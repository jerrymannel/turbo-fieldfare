# Porting plan: Qwen3.6-35B-A3B, and running two models on a Mac Mini M4

## Verdict up front

Your goal splits cleanly into two halves with very different difficulty:

- **Running two 35B-A3B MoE models concurrently on an M4 Mac Mini: genuinely
  feasible, and this runtime is close to the ideal architecture for it.** The
  resident cost is ~3.8 GB combined at 4K context, not 2 × 19 GB, because routed
  experts live on disk. The decode-service design already supports N instances.
- **Porting Qwen3.6-35B-A3B specifically: the MoE half is easy, the attention
  half is a rewrite.** Qwen3.6 uses Gated DeltaNet linear attention on 30 of its
  40 layers. That is not softmax attention, it does not use a KV cache, and this
  runtime has no equivalent. That single fact is most of the work.

The rest of this document quantifies both claims and proposes a staging that
de-risks the concurrency goal *before* paying for the DeltaNet port.

---

## 1. The target model

Qwen3.6-35B-A3B (Alibaba, released 15 April 2026, Apache 2.0). Published
architecture:

| Property | Value | vs. Gemma 4 26B-A4B |
| --- | --- | --- |
| Hidden size | 2,048 | 2,816 |
| Layers | 40 | 30 |
| Layer mix | 30 Gated DeltaNet + 10 full attention (every 4th: 3, 7, … 39) | 25 sliding-window + 5 full attention |
| Routed experts | 256, top-8 | 128, top-8 |
| Shared expert | 1, sigmoid-gated | 1, ungated |
| MoE intermediate | 512 | 704 |
| Attention heads | 16 Q / 2 KV (GQA 8:1), head_dim 256 | 16 Q / 8 KV, head_dim 256 |
| Vocab | 248,320 | 262,144 |
| Tied embeddings | **No** | Yes |
| Activation | SwiGLU | GeGLU (`gelu_pytorch_tanh`) |
| RoPE | mRoPE (3D), 25% partial rotary | NeoX, 25% partial rotary |
| Logit softcap | None | 30.0 |
| Context | 262K native | 128K |
| Extras | QK-Norm, gated attention, optional MTP head, vision encoder | QK-Norm, vision tower (omitted) |

> **Verify before building.** These come from published summaries, not from the
> actual `config.json`. Pull the real config from
> [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and
> confirm every number, especially the shared-expert intermediate size, the
> DeltaNet head/state dimensions, and the conv1d width. Also confirm an MLX
> 4-bit group-64 affine conversion actually exists — the installer's whole
> bounded-repack design assumes that source format.

---

## 2. Memory math: why two instances fit

### 2.1 Derived sizes

Routed expert parameters: `40 layers × 256 experts × 3 matrices × (2048 × 512)`
= **32.21 B parameters**, i.e. ~92% of the model is routed experts. This is the
number that makes the whole thing work — it is the part that stays on disk.

At MLX 4-bit group-64 affine (4 bits + BF16 scale + BF16 bias per 64 weights =
**4.5 bits/weight = 0.5625 bytes/param**):

| Component | Size | Where it lives |
| --- | ---: | --- |
| Routed experts (`packed_experts/`) | **18.12 GB** | Disk. Only selected blobs enter slots. |
| Common weights (`model_weights.bin`) | **~1.2 GB** (est.) | `mmap`ed, wrapped by Metal, no heap copy |
| **Total install** | **~19.3 GB** | vs. Gemma's 14.3 GB |

Per-expert blob: `3 × 2048 × 512 × 0.5625` = **1,769,472 bytes** — which is
exactly 108 × 16 KiB, so it is natively page-aligned with zero padding waste.
That is 1.9× smaller than Gemma's 3,358,720-byte blob, which makes every cache
miss correspondingly cheaper.

Expert slot capacity at the default 16 slots/layer:
`16 × 40 × 1,769,472` = **1.13 GB** — *less* than Gemma's 1.61 GB, despite 10
more layers, because the experts are smaller.

KV cache: only 10 layers have one, and GQA is aggressive (2 KV heads × 256 dim).
`10 × 2 × 2 × 256 × 2 bytes` = **20 KiB/token**:

| Context | Qwen3.6 KV | Gemma 4 KV |
| --- | ---: | ---: |
| 4K | 84 MB | 320 MB |
| 32K | 671 MB | — |
| 262K | 5.37 GB | — |

The 30 DeltaNet layers keep a **fixed-size recurrent state instead of a growing
cache** — O(1) in sequence length. Estimated at tens of MB total; confirm from
the real config. This is why long context is cheap on this model and why the KV
column above only counts 10 layers.

### 2.2 Two instances on a 16 GB M4 Mac Mini

The best case is **the same model twice**, because `ResidentBuffer` maps
`model_weights.bin` with `PROT_READ, MAP_PRIVATE`. A read-only private mapping
never dirties a page, so both processes share **one physical copy** of the
common weights via the unified buffer cache. The expert file cache is shared
too — instance B's misses are frequently served from RAM that instance A already
warmed.

| Item | Same model ×2 | Two different models |
| --- | ---: | ---: |
| Common weights (mmap) | 1.2 GB (shared) | 2.4 GB |
| Expert slots (private) | 2.26 GB | 2.26 GB |
| KV @ 4K | 0.17 GB | 0.17 GB |
| DeltaNet state (est.) | 0.10 GB | 0.10 GB |
| Scratch | 0.04 GB | 0.04 GB |
| **Total @ 4K** | **≈ 3.8 GB** | **≈ 5.0 GB** |
| **Total @ 32K** | **≈ 4.9 GB** | **≈ 6.1 GB** |

Against 16 GB with macOS taking ~3–4 GB, that leaves roughly 8 GB free — which
is not waste, it is exactly what you want, because free RAM becomes the file
cache that serves expert reads. A 24 GB M4 Mini would be comfortable at 32K on
both instances.

**Disk** is the more mundane constraint: ~19.3 GB for one model, ~38.6 GB for
two different ones. Fine on any M4 Mini, but worth noting against a base 256 GB
SSD.

### 2.3 The real bottleneck is SSD bandwidth, not RAM

Worst case, zero cache hits: `40 layers × 8 experts × 1.77 MB` = **566 MB per
generated token**. (Gemma's equivalent is 806 MB, and this repo sustains
5.1–6.3 tok/s on an 8 GB M2 and 31–35 tok/s on a 24 GB M5 Pro — see
[BENCHMARKS.md](BENCHMARKS.md). So the LFU hit rate must already be doing heavy
lifting; the worst case is not the observed case.)

Two concurrent instances contend for one SSD queue. Expect meaningfully less
than 2× aggregate throughput. Two points matter:

- **Run the same model twice, not two different models**, if your use case
  permits it. Shared page cache is the difference between the two instances
  helping each other and thrashing each other.
- The base 256 GB M4 Mini SSD is materially slower than higher-capacity configs
  (fewer NAND dies). If you have a choice of machine, capacity buys read
  bandwidth here, which converts directly into tokens/sec.

**This is the number I would measure before writing any port code** — see Stage 1.

---

## 3. What ports easily

The runtime is far less Gemma-welded than it first looks.

**Kernels are already dimension-parameterized.** The Metal shaders take shapes
via function constants with runtime-buffer fallbacks —
`FC_ROUTER_NUM_EXPERTS` (40), `FC_ROUTER_TOP_K` (42), `FC_MOE_TOP_K` (2),
`FC_ATTN_HEAD_DIM` (60), `FC_ATTN_NUM_Q_HEADS` (61), `FC_ATTN_NUM_KV_HEADS`
(62), and so on. **256 experts instead of 128 needs no kernel change at all.**
Top-8 is unchanged, so even the `router_topk_select_k8` specialization survives.

**`ArchConfig` is already a plain parameterized struct**
([ModelTypes.swift](../Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift)),
and the installer's `ArchInfo.load(configPath:)` already parses `config.json`.
The Gemma-specific part is only the `gemma4_26B_A4B` static and the field-by-field
equality gate.

**The entire streaming layer is architecture-agnostic.** `PreadExpertStreamer`,
the 16-slot LFU cache, page-aligned blobs, `MTLBuffer` wrapping, the bounded
range repack, manifest verification — none of this knows what a transformer is.
It moves fixed-stride byte ranges. It ports for free.

**A shared expert exists in both models**, so the parallel shared/routed branch
structure survives. Qwen's is sigmoid-gated where Gemma's is ungated — one extra
elementwise multiply.

---

## 4. What needs real work

### 4.1 Medium: the model-family refactor

| Item | Work |
| --- | --- |
| Arch validation | `Model.load` compares field-by-field against one static `ArchConfig`. Needs a model-family registry. Touches [ManifestReader](../Sources/TurboFieldfare/Infrastructure/ModelIO/ManifestReader.swift), [Model.swift](../Sources/TurboFieldfare/Runtime/Inference/Model.swift), `AppContextLengthOption`, `AppModelInstallationProbe`, `SupportedModelSource`. |
| Tensor naming | [RepackPlanner](../Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift) hard-codes `language_model.model.*`, `.router.proj.weight`, `.router.scale`, `.router.per_expert_scale`. Qwen uses `model.layers.N.mlp.experts.*` and `mlp.gate.weight`. Needs a name-mapping indirection. |
| SwiGLU vs GeGLU | The routed MoE kernel fuses GeGLU. Add a SwiGLU variant behind a function constant. |
| Untied embeddings | `tieWordEmbeddings` is already an `ArchConfig` field, but `LMHeadChainInt4` assumes the tied 4-bit head. Qwen needs a separate output-head tensor and its own resident budget (+0.29 GB). |
| Router quantization | Gemma's router is INT8 here. MLX Qwen conversions often leave the router gate in BF16. Needs an unquantized-router path. |
| Logit softcap | Config-driven already; set to 0 and skip the op. |
| mRoPE | 3D position encoding vs. Gemma's NeoX. For text-only it largely degenerates to 1D, but the section layout and 25% partial rotary need care. |
| Gated attention | Learned gate applied post-softmax on the 10 full-attention layers. Small addition. |

None of this is research. It is a few weeks of careful plumbing with tests.

### 4.2 Hard: Gated DeltaNet

This is the blocker, and it is worth being blunt about it. **30 of 40 layers use
a mechanism this runtime has no version of.** Gated DeltaNet is linear attention
with a delta-rule recurrence, roughly `S_t = (1-β)S_{t-1} + β·v·kᵀ`, plus a
causal conv1d (kernel width 4) for local mixing, plus gating.

What that requires:

1. **New Metal kernels** — causal conv1d, the recurrent state update, the gating,
   and the output projection path. Nothing existing is reusable; the attention
   kernels compute softmax over a KV window, which is a different operation.
2. **A new state manager.** `KVCacheManager`'s `LayerKind { swa, full }` becomes
   `{ deltaNet, full }`. The SWA ring machinery goes unused for Qwen (it has no
   sliding-window softmax layers), and is replaced by fixed-size per-layer
   matrix state.
3. **A chunkwise-parallel formulation for prefill.** The sequential recurrence is
   fine for decode (one token at a time is exactly its natural form) but would
   destroy prefill throughput. Linear-attention prefill needs the chunked
   parallel form, which is its own implementation and its own numerical-stability
   problem.
4. **CPU reference implementations and tolerance tests**, per this repo's
   established norm for every numerical kernel.

Realistically this is comparable in size to the existing attention + KV cache +
prefill-attention code *combined*. It is the majority of the port.

One consolation: DeltaNet is *cheaper* than attention at inference, has O(1)
state, and interacts well with expert streaming. The M4's memory bandwidth is
better spent on expert reads than on attention. If it works, it works well.

### 4.3 Operational: two instances is currently a policy ban, not a code limit

[AGENTS.md](../AGENTS.md) and
[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md#correctness-and-safety-invariants) both say
one model process at a time. That is an **8 GB validation-host rule**, not an
architectural constraint.

The code already supports N instances: `DecodeServiceInferenceClient` launches
each service via `launchctl` with a unique label and a per-instance socket at
`/private/tmp/turbofieldfare-decode-<identifier>.sock`. Two decode services, or
two `TurboFieldfareServer` processes on different ports, work today.

So "run 2" means: relax the operational rule for your machine, and document that
published benchmarks remain single-instance. `InstallLock` serializes install and
promotion per path but does not restrict concurrent readers.

---

## 5. Proposed staging

Ordered to put the cheapest risk-reduction first.

**Stage 1 — Prove the concurrency goal with the model you already have.**
No porting. Run two `TurboFieldfareServer` instances on different ports against
the existing `scratch/gemma4.gturbo`, on the M4 Mini, and measure: aggregate
tokens/sec, per-instance decode rate, peak footprint, I/O per token, and whether
page-cache sharing materializes. This answers the *actual* question — "can this
Mac run two streamed MoE models at a useful speed?" — in a day, using Gemma as
the stand-in. If the answer is no, the port is moot. Requires relaxing the
AGENTS.md single-process rule; keep it relaxed only for your own experiments.

**Stage 2 — Model-family plumbing, Gemma still passing.**
Registry, name mapping, per-family `ArchConfig`, SwiGLU kernel variant, untied
head, unquantized-router path. Ship it with Gemma still green in CI. Zero new
math, and it is the prerequisite for anything else.

**Stage 3 — Port a pure-softmax MoE first.**
A conventional Qwen3-family MoE (softmax attention throughout) shakes out tensor
naming, SwiGLU, untied embeddings, and router quantization against a real second
model, without touching DeltaNet. This is the highest-value intermediate
milestone: it turns "multi-model runtime" from a claim into a fact.

**Stage 4 — Gated DeltaNet.**
Kernels, state manager, chunkwise-parallel prefill, reference tests. Budget this
as the largest single piece of work in the project.

**Stage 5 — Qwen3.6 specifics.** mRoPE, gated attention, the 262K context path.
Optionally the MTP head for self-speculative decoding, which is interesting here
because it trades compute (cheap, idle) for expert reads (expensive, the
bottleneck).

---

## 6. Open questions to resolve first

1. **Does an MLX 4-bit group-64 affine conversion of Qwen3.6-35B-A3B exist?** The
   installer's bounded repack is built around that exact source format. Without
   it you are also writing a quantizer, which is a separate project.
2. **What are the real DeltaNet dimensions** (heads, state size, conv width)?
   These set the state-memory number I estimated at ~0.1 GB.
3. **Same model twice, or two different models?** This changes resident cost by
   ~1.2 GB and, more importantly, changes whether the two instances share page
   cache or fight over it.
4. **Which M4 Mini?** RAM decides your context ceiling; SSD capacity decides your
   read bandwidth, which decides tokens/sec.
5. **What are the two instances actually for?** If it is parallel agents, two
   servers is right. If it is draft-and-verify speculative decoding, the MTP head
   in Stage 5 may be a better answer than a second process.

---

## Sources

- [Qwen/Qwen3.6-35B-A3B — Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6-35B-A3B — Architecture Overview](https://huggingface.co/blog/EXDai/qwen36-35b-a3b-architecture-overview)
- [Qwen3.6 35B A3B: Specifications and GPU VRAM Requirements — apxml](https://apxml.com/models/qwen36-35b-a3b)
