# TurboFieldfare — Repository Analysis

A developer's orientation guide, written to get you from "new to this repo" to
"ready to build a feature."

**What this project is:** a from-scratch Swift + Metal inference runtime for the
Gemma 4 26B-A4B (MoE) instruction model, designed to run a ~14.3 GB model on an
8 GB Apple Silicon Mac by streaming routed experts from disk instead of
resident-loading them. It is not a wrapper around llama.cpp or MLX — the
transformer, the quantized GEMV kernels, the KV cache, the sampler, the model
installer, and the OpenAI-compatible server are all in this repo.

**Platform floor:** macOS 26 / iOS 26, Swift 6.2, Metal 4 (MSL 4.0). These are
hard requirements, not preferences — the prefill path uses Metal Performance
Primitives tensor ops that only exist in MSL 4.0.

---

## 1. The packages used

### External Swift package dependencies

Only two are declared directly in [Package.swift](Package.swift):

| Package | Version | Used for | Where |
| --- | --- | --- | --- |
| `huggingface/swift-transformers` | from 1.3.0 (resolved 1.3.3) | The `Tokenizers` product only — BPE encode/decode and the Jinja chat template | [Tokenizer.swift](Sources/TurboFieldfare/Tokenization/Tokenizer.swift), [Detokenizer.swift](Sources/TurboFieldfare/Tokenization/Detokenizer.swift) |
| `apple/swift-nio` | **exact** 2.99.0 | `NIOCore` / `NIOPosix` / `NIOHTTP1` for the loopback HTTP server; `NIOEmbedded` in tests | [Sources/TurboFieldfareServer/Core/](Sources/TurboFieldfareServer/Core/) |

Note that swift-nio is pinned with `exact:`, not `from:`. Don't loosen it
casually — it was pinned deliberately.

Everything else in [Package.resolved](Package.resolved) is transitive, pulled in
by swift-transformers: `swift-huggingface`, `swift-jinja`, `swift-collections`,
`swift-crypto`, `swift-asn1`, `swift-atomics`, `swift-system`, `EventSource`,
`yyjson`. You should not import these directly; if you need one, add it as an
explicit dependency rather than relying on it leaking through.

### System frameworks

- **Metal** — everywhere in the runtime. `MTLBuffer`, `MTLComputeCommandEncoder`,
  `makeBuffer(bytesNoCopy:)` for zero-copy weight wrapping.
- **MetalPerformancePrimitives** — included from
  [tensorops.metal](Sources/TurboFieldfare/Metal/TensorCore/tensorops.metal) for
  the `matmul2d` prefill path. This is why MSL 4.0 is mandatory.
- **SwiftUI / AppKit** — the Mac app (`Sources/TurboFieldfareApp/Mac/`).
- **Foundation / Darwin** — POSIX `pread`, `mmap`, `fcntl`/`F_RDADVISE`, advisory
  locks. See [Posix.swift](Sources/TurboFieldfareRepack/Core/System/Posix.swift).

### Tooling

- **swift-testing** (`import Testing`, `@Test`, `#expect`) — all 127 test files.
  There is zero XCTest in this repo. Match that.
- **Ruby** — [Scripts/check_markdown_links.rb](Scripts/check_markdown_links.rb),
  run in CI. Broken doc links fail the build.
- **GitHub Actions** on `macos-26`: release build → serial tests → markdown link
  check. See [ci.yml](.github/workflows/ci.yml). CI never downloads or loads the
  real model.

### Notably absent

No numerical/ML library (no Accelerate BLAS, no MPSGraph, no MLX). No JSON
library — hand-rolled [JSONValue.swift](Sources/TurboFieldfare/Tokenization/JSONValue.swift).
No argument parser (not even swift-argument-parser) — CLI parsing is hand-written
in [Args.swift](Sources/TurboFieldfareCLI/Args.swift) and
[ServerArguments.swift](Sources/TurboFieldfareServer/Core/ServerArguments.swift).
No logging framework. This minimalism is intentional and you should preserve it.

---

## 2. Architecture

### 2.1 Target graph

Six products, twelve targets. The consistent pattern is **`Core` library target +
thin `Command` executable target**, which exists so the logic is testable — an
executable target can't be a test dependency.

```
                    TurboFieldfare  (the runtime library)
                    ├── Tokenizers (swift-transformers)
                    └── Metal shaders as bundle resources
                             │
        ┌────────────────────┼────────────────────┬──────────────────┐
        │                    │                    │                  │
 TurboFieldfareCLICore  TurboFieldfareAppCore  TurboFieldfareServerCore   TurboFieldfareValidationSupport
        │              ├── RepackCore          └── swift-nio              (reference impls for tests)
        │              └── DecodeProtocol              │
        │                    │                         │
 TurboFieldfareCLI    ┌──────┴───────┐          TurboFieldfareServer
   (executable)       │              │             (executable)
              MacPresentation   DecodeService
                      │         (executable)
              TurboFieldfareMac
                (executable)

 TurboFieldfareRepackCore ──> TurboFieldfareRepack (executable, the installer)
```

Target-by-target:

| Target | Path | Role |
| --- | --- | --- |
| `TurboFieldfare` | `Sources/TurboFieldfare/` | The inference runtime: model load, KV cache, kernels, Metal shaders, tokenization, generation loop |
| `TurboFieldfareRepackCore` | `Sources/TurboFieldfareRepack/Core/` | Model installer: streams from HuggingFace, repacks into `.gturbo`, verifies |
| `TurboFieldfareCLICore` | `Sources/TurboFieldfareCLI/` | Arg parsing + run loop for the CLI |
| `TurboFieldfareAppCore` | `Sources/TurboFieldfareApp/Core/` | Platform-agnostic app logic: state, install orchestration, inference clients, diagnostics |
| `TurboFieldfareMacPresentation` | `Sources/TurboFieldfareApp/MacPresentation/` | Markdown rendering, theme, transcript export — testable presentation logic |
| `TurboFieldfareMac` | `Sources/TurboFieldfareApp/Mac/` | SwiftUI views only |
| `TurboFieldfareDecodeProtocol` | `Sources/TurboFieldfareDecodeProtocol/` | Length-prefixed JSON IPC types shared by app and decode service |
| `TurboFieldfareDecodeService` | `Sources/TurboFieldfareDecodeService/` | Out-of-process model host for the Mac app |
| `TurboFieldfareServerCore` | `Sources/TurboFieldfareServer/Core/` | OpenAI-compatible HTTP server on NIO |
| `TurboFieldfareValidationSupport` | `Sources/TurboFieldfareValidation/Support/` | Deterministic PRNG, FP16 buffers, CPU reference implementations, tolerance helpers — **test-only support, not shipped logic** |

### 2.2 Why the Mac app runs the model out-of-process

The Mac app does **not** load the model in-process. It spawns
`TurboFieldfareDecodeService` as a sibling executable and talks to it over a Unix
socket with length-prefixed JSON frames
([DecodeProtocol.swift](Sources/TurboFieldfareDecodeProtocol/DecodeProtocol.swift),
4 MiB max payload).

This buys three things: the ~1.6 GB resident weights plus expert slots live in a
separate address space from SwiftUI (so the HUD reports honest memory numbers for
the model process); a runtime crash doesn't take the UI down; and the app can
unload/reload the model without leaking Metal state into a long-lived UI process.

The consequence for you: **any new runtime knob the app needs to control must be
plumbed through the IPC types**, not just passed as a Swift value. That's a
4-place edit — see §3.2.

### 2.3 The model format (`.gturbo`)

Read [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the authoritative version.
In short:

```
gemma4.gturbo/
  manifest.json          # architecture, sizes, SHA-256 — absence means "partial install"
  verified-install.json  # which manifest/dir/files were verified (trusted-receipt mode)
  model_weights.bin      # ~1.35 GB: embedding/tied head, attention, routers, shared experts, norms
  tokenizer/             # config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
  packed_experts/
    layout.json          # packed sub-region offsets within each expert blob
    layer_00.bin … layer_29.bin   # 128 fixed-stride routed experts per layer, ~12 GiB total
```

`model_weights.bin` is `mmap`ed read-only and wrapped in `MTLBuffer`s with no heap
copy. Expert layer files are opened lazily; each opened layer owns one fd and 16
page-aligned slot buffers filled by bounded `pread` on a cache miss (LFU eviction,
recency tiebreak).

The installer never materializes a full shard: it reads the pinned HF index,
plans byte ranges, and copies quantized values through ≤512 KiB scratch, straight
into final destinations. Quantized values are copied **unchanged** — never
dequantized and requantized.

### 2.4 The two execution modes

**Prefill** (up to 128 prompt tokens per chunk) is layer-major: each bounded row
group moves through the transformer one layer at a time. 4-bit projections
dequantize a tile into FP16 threadgroup memory and hand it to MPP `matmul2d`.
Routed MoE is grouped into tiles of ≤8 experts so the next tile can be fetched
while the current tile's GPU work is queued — both fit in the 16-slot cache.

**Decode** is one token at a time, split across three phases per layer:

| Phase | Work |
| --- | --- |
| `cb1` | norm, QKV, RoPE, KV write, attention, O projection, router → top-8 expert IDs ready for CPU readback |
| `io` | CPU plans cache hits/misses, `pread`s missing experts in parallel; the resident shared-expert branch runs on GPU concurrently |
| `cb2` | routed top-8 branch, router-weight reduction, combine with shared branch, layer tail |

The CPU↔GPU handoff in the middle is the structural reason decode is I/O-shaped:
you cannot know which experts to read until the GPU tells you.

Entry points: [runRawCompletion](Sources/TurboFieldfare/Runtime/Generation/RawCompletion.swift)
owns the outer loop; [RealForwardRunner](Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift)
(1,785 lines — the biggest file in the repo) owns the per-layer graph for both modes.

### 2.5 The abstraction seam: `LogitProducer`

[LogitProducer.swift](Sources/TurboFieldfare/Runtime/Generation/LogitProducer.swift)
defines the only real polymorphism in the runtime:

- `LogitProducer` — `produce(token:position:into:)`, `reset()`
- `ContinuableLogitProducer` — supports KV-prefix resume (used by the server's prompt cache)
- `ChunkedPrefillRunner` — supports the chunked prefill path
- `ContextWindowReporting` — reports `maxContext`

`runRawCompletion` does runtime `as?` checks against these to decide its path.
`RealForwardRunner` implements all of them; test doubles like `ScriptedLogitProducer`
implement the minimum. **This is the seam to use if your feature needs to intercept
or wrap generation.**

### 2.6 The head-path fork (a real gotcha)

There are two output modes after layer 30:

- **Fused greedy head** (default): the GPU computes argmax and returns a token.
  The logits buffer is *never written*. Fast, but only valid for `temperature == 0
  && repetitionPenalty == 1`.
- **Logits head** (`forceLogitsHead: true`): writes the full vocab logits so the
  sampler can run.

`runRawCompletion` throws if a sampling config meets a fused-greedy producer. The
choice is made **at model construction time**, which is why the docs say some
sampling changes require a reload. If your feature touches sampling, logit
biasing, or per-token inspection, you almost certainly need the logits path and
must handle the reload.

Sampling order is Top-P → Top-K → temperature; the default Top-K 64 uses a
specialized 1024→64 reduction.

### 2.7 The three front ends

| Front end | Prompt framing | Notes |
| --- | --- | --- |
| **CLI** | `--prompt` = raw completion, no chat template; `--messages-file` = pinned Gemma 4 IT format | Best surface for reproducible experiments |
| **Mac app** | Pinned Gemma 4 IT chat format, one user prompt | Out-of-process via decode service |
| **Server** | Full upstream Jinja template — developer messages, function declarations, tool calls, tool results | Loopback only, one warm model, serialized generation, single-prefix KV cache |

The server exposes `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
([HTTPServer.swift:170-193](Sources/TurboFieldfareServer/Core/HTTPServer.swift#L170)).
Native `<|tool_call>` blocks are parsed by
[GemmaToolCallParser](Sources/TurboFieldfare/Tokenization/GemmaToolCallParser.swift)
and fail closed on malformed output.

Stop tokens across all front ends: `<eos>` (1), `<turn|>` (106), `<|tool_response>` (50).

### 2.8 Metal shader pipeline

Shaders are **compiled at runtime, not build time**.
[MetalContext](Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift)
concatenates ten `.metal` files from the bundle into one source string and calls
`makeLibrary(source:options:)` with `languageVersion = .version4_0`. Pipelines are
cached by `(name, function constants, threadgroup hint)`.

The upside is a fast dev loop: edit a shader, rebuild the Swift target, done — no
Xcode metallib step. The downside is that **shader syntax errors are runtime
failures, not compile failures**, and any name collision across the ten modules
breaks the combined library.

`tensorops.metal` is deliberately *not* in the combined list — it is compiled
separately via `moduleLibrary(device:module:)` because of its MPP include.

---

## 3. How to add features

### 3.0 First, pick your layer

Ask which of these your feature actually is, because the file sets barely overlap:

| If your feature is… | Start here |
| --- | --- |
| A new generation control (sampler, penalty, stop behavior) | `Runtime/Generation/` + all three front ends |
| A new runtime/perf knob (cache policy, scheduling, I/O) | `RuntimeConfiguration` → `AppRuntimeOptions` → `DecodeRuntimeOptions` |
| A new HTTP endpoint or API field | `TurboFieldfareServerCore` only |
| A UI feature (transcript, export, presets) | `AppCore` + `MacPresentation` + `Mac` views |
| A new Metal kernel or fusion | `Metal/*.metal` + `Kernels/` wrapper + reference test |
| Installer / model format change | `RepackCore` + `ManifestReader` + format validators |

### 3.1 Adding a generation control (e.g. min-P, a new penalty, a new stop mode)

The type has to travel through every front end. Follow the existing `topP` field
as a template:

1. **`GenerationConfig`** in [Generator.swift](Sources/TurboFieldfare/Runtime/Generation/Generator.swift) — add the field + validation in `validate()`.
2. **[Sampler.swift](Sources/TurboFieldfare/Runtime/Generation/Sampler.swift)** and possibly [logit.metal](Sources/TurboFieldfare/Metal/Sampling/logit.metal) — implement it.
3. **CLI**: add the flag to `Args`, the parse case, the usage string, and wire it in [Run.swift](Sources/TurboFieldfareCLI/Run.swift).
4. **Server**: add the field to [OpenAIModels.swift](Sources/TurboFieldfareServer/Core/OpenAIModels.swift) with validation, then map it in [ServerInference.swift](Sources/TurboFieldfareServer/Core/ServerInference.swift).
5. **App**: `AppGenerationRequest` → `DecodeGenerationRequest` → decode service `Entry` → the SwiftUI control.
6. **Tests**: a sampler unit test with a `ScriptedLogitProducer`, an arg-parse test, an OpenAI-validation test.
7. **Docs**: the generation-controls table in [RUNTIME_CONTROLS.md](docs/RUNTIME_CONTROLS.md) and the CLI section of the README.

If your control makes greedy generation impossible, remember §2.6 — you may need
`forceLogitsHead`.

### 3.2 Adding a runtime knob (e.g. a new cache policy, a prefetch strategy)

This is the four-hop plumb:

```
RuntimeConfiguration            Sources/TurboFieldfare/Runtime/Configuration/
      ↑ resolvedRuntimeConfiguration()
AppRuntimeOptions               Sources/TurboFieldfareApp/Core/Configuration/
      ↕ encode/decode
DecodeRuntimeOptions            Sources/TurboFieldfareDecodeProtocol/
      ↕
DecodeService Entry             Sources/TurboFieldfareDecodeService/
```

Two things to get right:

- `RuntimeConfiguration` validates with `precondition` (crashes on bad input),
  while `AppRuntimeOptions.validate()` throws. Keep enumerated allowed-value lists
  in `RuntimeConfiguration` (`allowedExpertCacheSlots`, `allowedPrefillChunkTokens`)
  and have the app layer reference them — don't duplicate the list.
- `DecodeRuntimeOptions` uses `String` for enums, not the enum types, so the IPC
  boundary stays version-tolerant. Follow that convention.
- Decide whether your knob is load-time or per-request. Load-time knobs must be in
  `AppLoadedRuntimeKey`, which is what triggers a model reload when it changes.
  Getting this wrong means the setting silently doesn't apply.

### 3.3 Adding a server endpoint or API field

Fully contained in `TurboFieldfareServerCore`:

1. Add the route case in `route(head:...)` in [HTTPServer.swift](Sources/TurboFieldfareServer/Core/HTTPServer.swift) — note the existing catch-all that returns 405 for known paths with the wrong method.
2. Add request/response `Codable` types to [OpenAIModels.swift](Sources/TurboFieldfareServer/Core/OpenAIModels.swift) with explicit validation and OpenAI-shaped errors.
3. Tests go in `Tests/TurboFieldfareServer/` using `NIOEmbedded` and a scripted `ServerInferenceBackend` actor — the existing `HTTPServerTests` show the pattern, and no real model is needed.
4. Document it in [docs/OPENAI_SERVER.md](docs/OPENAI_SERVER.md).

`ServerInferenceBackend` is a protocol, which is exactly why the server is testable
without a model. Keep it that way — don't reach around it into `Model` directly.

### 3.4 Adding a Metal kernel

1. Write the kernel in the right `Sources/TurboFieldfare/Metal/<Area>/*.metal`. If you add a *new file*, register it in **both** `shaderModules` and `shaderSubdirectories` in [MetalContext.swift](Sources/TurboFieldfare/Infrastructure/Metal/MetalContext.swift), and add the resource copy rule if the directory is new.
2. Add a Swift wrapper in `Sources/TurboFieldfare/Kernels/<Area>/` that owns the pipeline lookup and encoding. Existing wrappers are the style guide.
3. **Write a CPU reference implementation** in `Sources/TurboFieldfareValidation/Support/Reference/` and a tolerance-based test. This is the established norm — see the 9 files under `Tests/TurboFieldfare/Core/Validation/Reference/`. Use `SplitMix64`/`SeedTree` for deterministic inputs.
4. Watch alignment: routed gate/up projections may be only 2-byte aligned and build 4-byte values from two `ushort` loads; down-projection offsets are 4-byte aligned and can use `uint` loads. Wider loads are only valid where alignment is guaranteed.

### 3.5 Adding a UI feature

Put logic in `TurboFieldfareAppCore` or `TurboFieldfareMacPresentation` (both have
test targets); keep `Sources/TurboFieldfareApp/Mac/` to SwiftUI views only. State
lives in [AppModel.swift](Sources/TurboFieldfareApp/Core/State/AppModel.swift) (842
lines, the app's hub) with presentation-derived values in `AppPresentationState`.
Prompt presets are data in [app-prompts.json](Sources/TurboFieldfareApp/Core/Resources/app-prompts.json),
not code.

### 3.6 What a real feature commit looks like here

Two recent ones, for calibration:

- **`e65da9e` — OpenAI server + tool calling**: +4,346 lines across 33 files. New target in `Package.swift`, 5 new server source files, tool-call parsing in the runtime, **5 new test files (~1,135 lines of tests)**, a new doc, plus updates to `README.md`, `AGENTS.md`, and `SYSTEM_DESIGN.md`.
- **`489f2d1` — slim resumable downloads**: touched 34 files across installer, app state, and UI, with tests and doc updates in the same commit.

The pattern is consistent: **source + tests + docs in one commit**, and the docs
are treated as part of the deliverable, not an afterthought.

---

## 4. What you should know before making changes

### 4.1 Read AGENTS.md first — it constrains this checkout

[AGENTS.md](AGENTS.md) opens with: *"This checkout is for running and reporting
existing behavior. Do not edit source, change runtime defaults, or start
optimization work unless the user asks."* Since you're asking for a feature, that
gate is open — but the operational rules in that file still apply:

- **Never run two model processes at once.** Check with
  `pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper'`
  before any model run.
- **Never terminate an existing model process or delete/reinstall the model.** The
  install is a ~15 GB download.
- Don't duplicate the `.gturbo` directory, create a worktree, or purge caches just
  to run tests.

There is a working install at `scratch/gemma4.gturbo` in this checkout. Treat it
as precious.

### 4.2 The invariants you must not break

From [SYSTEM_DESIGN.md § Correctness and safety invariants](docs/SYSTEM_DESIGN.md):

1. **No whole model, shard, expert, or source tensor in Swift heap memory** — ever, in any install, load, test, or benchmark path. This is the entire premise of the project. Use `mmap` + `MTLBuffer` wrapping or bounded `pread` into preallocated slots.
2. **Quantized values are preserved exactly.** The repacker changes layout, never numerics. Lossless repack changes require byte or output identity.
3. **K and V stay distinct** after their separate normalization and RoPE paths, even where the raw projection is shared (full-attention layers).
4. **Every queued GPU consumer owns its slot and scratch bank until completion.** CPU reuse must not race an earlier command buffer. If you add a scheduling optimization, this is the invariant you will be tempted to break.
5. **Kernels that reorder float ops must stay deterministic within a path** and within reference tolerances. Bit-exact identity across paths is not required.
6. Slow profiling modes are diagnostic evidence, not throughput evidence.

### 4.3 Concurrency: Swift 6 language mode is on

`swift-tools-version: 6.2` with no `swiftSettings` overrides means **strict
concurrency checking**. Nearly every public type is `Sendable`. Where it can't be,
the repo uses `@unchecked Sendable` **with a comment stating the actual ownership
contract** — see `MetalContext` ("device/queue/library immutable, pipeline cache
lock-guarded") and `RawCompletionScratch` ("exclusively owned by one generation;
the single-in-flight guard upstream is the contract").

If you add an `@unchecked Sendable`, write that comment. It's a house rule that's
been followed without exception.

### 4.4 Tests: serial, swift-testing, reference-validated

- **Always run via [Scripts/test.sh](Scripts/test.sh)**, which is
  `swift test --no-parallel`. Shared Metal state makes parallel in-process tests
  unreliable. Running `swift test` directly will give you flaky failures.
- 127 test files, **100% swift-testing** (`import Testing`). No XCTest.
- Test directory structure mirrors source structure exactly. Put a test for
  `Sources/TurboFieldfare/Kernels/MoE/X.swift` at
  `Tests/TurboFieldfare/Core/Kernels/MoE/XTests.swift`.
- Numerical code gets a CPU reference implementation and a tolerance comparison,
  not a golden-value assertion.
- CI runs on `macos-26` without a model, so **every test must pass without the
  model present**. If your feature needs the model to test meaningfully, structure
  it behind a protocol (as the server does with `ServerInferenceBackend`) so the
  logic is testable with a double.

The full pre-PR check:

```bash
swift build -c release && Scripts/test.sh && ruby Scripts/check_markdown_links.rb
```

### 4.5 Documentation is load-bearing

`ruby Scripts/check_markdown_links.rb` runs in CI, and the docs cross-reference
source files by path. **Moving or renaming a source file can break CI** via the
code-map links in [SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md).

The doc set and what each owns:

| Doc | Owns |
| --- | --- |
| [README.md](README.md) | User-facing install, CLI, app overview |
| [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) | Architecture, invariants, code map |
| [docs/RUNTIME_CONTROLS.md](docs/RUNTIME_CONTROLS.md) | Every user-visible knob and its default |
| [docs/OPENAI_SERVER.md](docs/OPENAI_SERVER.md) | Server API surface |
| [docs/OPTIMIZATION_JOURNEY.md](docs/OPTIMIZATION_JOURNEY.md) | Experiments, including reverted ones |
| [docs/experiments/EXPERIMENT_INVENTORY.md](docs/experiments/EXPERIMENT_INVENTORY.md) | Complete experiment index |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md) | Measurement protocol |
| [AGENTS.md](AGENTS.md) | Operational rules for this checkout |

**Before optimizing anything, read `OPTIMIZATION_JOURNEY.md` and the nine
experiment summaries under `docs/experiments/summaries/`.** They cover expert
cache prediction, RDADVISE, attention/KV, prefill, fusions, sampling — including
what was tried and reversed. A meaningful amount of the obvious perf ideas have
already been tested and rejected with data.

### 4.6 Explicit non-goals

Out of scope per SYSTEM_DESIGN: vision input (the vision tower is deliberately
omitted at repack time), training, fine-tuning, server batching, remote serving,
and general multi-model support. The server is loopback-only with no auth and no
TLS — don't proxy or tunnel it, and don't add "just a small" auth layer without
deciding that remote serving is now in scope.

Context lengths 4K–64K are offered; **published acceptance evidence only covers
4K**. If your feature interacts with long context, you're in under-validated
territory and should say so.

### 4.7 Practical friction to expect

- **The prompt-shape reload rule** trips people up. Load-time vs. per-request is
  the first design question for any new knob (§3.2).
- **Shader errors are runtime errors.** Budget for a rebuild-and-run cycle when
  touching `.metal`.
- **`RealForwardRunner.swift` is 1,785 lines** and owns both prefill and decode
  graphs. Most kernel-level features land here. Read it before you plan.
- **Hand-rolled arg parsing** means adding a flag is three edits (parse case,
  struct field, usage string) with no compiler enforcement that you did all three.
  There are arg-parse tests — extend them.
- The repo has **no linter or formatter config**. Match surrounding style by
  reading neighbors, especially comment density: this codebase comments the *why*
  (ownership contracts, alignment constraints, hardware rationale) and not the *what*.

---

## 5. Suggested next step

Since your goal is a new feature, the highest-value thing you can do before
writing code is decide **which layer it lives in** (§3.0) and **whether it is
load-time or per-request** (§3.2). Those two answers determine your entire file
set and whether you need model reload handling.

If you tell me what the feature is, I can map it to the concrete files and give
you an implementation plan against this structure.
