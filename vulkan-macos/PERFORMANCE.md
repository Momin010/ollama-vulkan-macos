# Characterising LLM inference on Intel Macs with discrete AMD GPUs

**Momin Aldahdouh** · July 2026 · [ollama-vulkan-macos](https://github.com/Momin010/ollama-vulkan-macos)

## Abstract

Ollama and llama.cpp have no supported GPU path on Intel Macs with discrete
AMD GPUs: Metal support targets Apple Silicon, and inference silently falls
back to the CPU. We route inference through Vulkan via MoltenVK and
characterise the result on an AMD Radeon Pro 5500M, reaching 45.7 tok/s on
Llama 3.2 3B Q4_K_M against a 13.1 tok/s CPU baseline, a 3.5x speedup.

We then attempt to close the remaining gap to the hardware's rated 192 GB/s of
memory bandwidth, and report mostly negative results: no environment
configuration, no architecture-detection fix, and no speculative decoding
scheme using a draft model improves generation throughput. Native Metal, the
obvious alternative to a translation layer, is 20x *slower*.

The central result is that the rated bandwidth is not a meaningful denominator.
A minimal streaming-read kernel measures the GPU's actual ceiling at
**141.4 GB/s, 74% of its rated figure**. Against that ceiling, inference
already achieves 65% (Q4_K_M) to 83% (f16). We then test the obvious explanation for the
shortfall at 4-bit -- dequantisation cost -- by quantising one model into four
4-bit formats with very different unpacking schemes, and **refute it**: all
four land within 55-61% regardless. At 4-bit the kernel saturates neither
bandwidth (60%) nor arithmetic (~8% of fp32 peak), which points to latency or
occupancy in the matrix-vector reduction instead.

One positive result: n-gram cache speculative decoding improves throughput by
19% on inputs whose output repeats their context, at no bandwidth cost.

## 1. Background

Token generation for a dense transformer is memory-bandwidth-bound. Each token
requires reading every weight once, so the throughput ceiling is

    tokens/s ≤ achievable_bandwidth / model_size_in_bytes

For Llama 3.2 3B Q4_K_M (2.02 GB) on a GPU rated at 192 GB/s, that suggests
95 tok/s. We measured 45.7 — 48% — and set out to find the missing half.

**Hardware.** 2019 16" MacBook Pro; AMD Radeon Pro 5500M, 8 GB GDDR6, 128-bit
bus at 12 Gbps = 192 GB/s rated; Intel UHD 630 integrated; macOS with
MoltenVK 1.4.1 and Vulkan loader 1.4.350.1.

## 2. Method

Every measurement uses a freshly launched `llama-server`, verified via `lsof`
to be the process listening on the benchmark port, with a warm-up request
discarded and cooldowns between configurations. Runs are bracketed by repeated
baseline measurements so drift is visible rather than silent.

This rigour was not optional. Our first sweep was **entirely invalid**: the
cleanup used `pkill -f "llama-server --port N"`, which never matches, because
the real command line is `llama-server --model … --port N`. Servers survived,
subsequent configurations failed to bind the port and exited, and the health
check answered from the *first* server. Seven "different configurations"
agreed within 4% because they were one configuration measured seven times.
The bug is trivial; the failure mode — plausible, self-consistent, entirely
fictitious data — is not. **Verify that the process answering you is the
process you started.**

## 3. Negative results

### 3.1 Environment configuration (null)

ggml's Vulkan backend exposes 34 environment knobs. Six plausible candidates
were tested: f16 matmul, forced and disabled MMVQ, integer dot product,
submission batching. All landed within a 2.6% spread, and the two baseline
runs bracketed the entire range. No knob affects generation throughput.

### 3.2 AMD architecture detection (real bug, no effect)

`get_device_architecture()` identifies AMD GPUs via `VK_AMD_shader_core_properties`.
MoltenVK does not expose it, so **every AMD GPU on macOS is classified as
`OTHER`** and loses its architecture-specific pipeline tuning. For RDNA1 that
table specifies `{"mul_mat_vec", 64}` — the generation kernel at subgroup
size 64 rather than the RDNA default of 32.

We restored the classification using the subgroup-size signature the device
does report (`min 32 / max 64`) and rebuilt. Throughput was unchanged
(44.85 vs 45.33 tok/s). MoltenVK already defaults `subgroupSize` to 64, so the
tuning it would have restored was already in force.

The misclassification is real and worth fixing upstream for correctness; it is
not a performance defect on this path.

### 3.3 Native Metal (20x worse)

Bypassing the translation layer is the obvious hypothesis. We built pristine
llama.cpp with `GGML_METAL=ON` for x86_64. It ran correctly — device selected,
29/29 layers offloaded, `simdgroup reduction = true` — at **2.34 tok/s**.

ggml's Metal kernels assume unified memory and `simdgroup_matrix` operations
(`MTLGPUFamilyApple7`), neither of which a discrete AMD GPU provides. Vulkan
through MoltenVK is not a handicap on this hardware; it is dramatically the
better path, and the translation layer is not where the bandwidth goes.

### 3.4 Speculative decoding with a draft model (halves throughput)

Speculative decoding amortises one weight pass over several tokens, which in
principle breaks the one-token-per-pass ceiling. It fails here for an
arithmetic reason: the smallest same-tokenizer draft for Llama 3.2 is the 1B at
1.32 GB, against a 2.02 GB target. One cycle at depth 5 reads more than
break-even requires, and measured throughput fell to 20.25 tok/s.

`ngram-simple` and `ngram-mod` also lost (20–42 tok/s). Batched verification is
expensive on this GPU — prefill runs at only 35–52 tok/s, an order of magnitude
below expectation — so schemes that trade batch work for weight traffic are
poorly matched to it.

*(We initially recorded a null rather than a loss here, because `--spec-type`
defaults to `none` and our first attempt set only `--spec-draft-model`.
Speculative decoding never ran. Corrected.)*

## 4. The rated bandwidth is the wrong denominator

Every efficiency figure above is computed against 192 GB/s. That number is a
bus rate, not a capability. We measured the GPU directly with a Metal kernel
that streams a 1 GiB private buffer and accumulates — pure coalesced sequential
read, no dequantisation, a strict upper bound on any inference kernel:

    BEST STREAMING READ: 141.4 GB/s   (74% of rated 192 GB/s)

Recomputed against what the hardware can actually deliver:

| configuration | bytes/weight | tok/s | GB/s | % of 192 | **% of 141.4** |
|---|---|---|---|---|---|
| 3B Q4_K_M | 0.56 | 45.68 | 92.3 | 48% | **65%** |
| 1B Q8_0 | 1.06 | 77.69 | 102.6 | 53% | **73%** |
| 3B Q8_0 | 1.06 | 33.06 | 113.1 | 59% | **80%** |
| 1B f16 | 2.00 | 47.55 | 117.9 | 61% | **83%** |
| *streaming read* | — | — | *141.4* | *74%* | *100%* |

Inference is not running at half the hardware's capability. It is running at
65–83% of it. The apparent shortfall was mostly an artefact of dividing by a
number no kernel on this GPU can reach.

## 5. Where the remaining gap is

The efficiency ordering is monotonic in bytes per weight. f16 — raw read, no
unpacking — reaches 83%; Q4_K_M reaches 65%. The obvious reading is that the
gap is dequantisation ALU cost, and an earlier draft of this report claimed
exactly that, along with a 28% prize for removing it.

**That claim was wrong, and the experiment that killed it is below.**

If unpacking cost were the limiter, 4-bit formats with very different unpacking
schemes should differ in throughput. We quantised one Llama 3.2 1B f16 file
into four 4-bit formats — identical weights, identical shapes, only the packing
scheme varies — and measured each on a fresh verified server:

| format | unpacking work | GB/s | % of 141.4 | G weights/s | tok/s |
|---|---|---|---|---|---|
| Q4_0 | `d × (x − 8)` | 85.4 | 60% | 152 | 110.73 |
| Q4_K_M | super-block, 6-bit scales *and* mins | 86.1 | 61% | 153 | 106.61 |
| Q4_1 | scale + min per block | 84.0 | 59% | 149 | 101.01 |
| IQ4_NL | non-linear lookup table | 77.9 | 55% | 138 | 100.17 |
| *Q8_0* | *`d × x`* | *102.6* | *73%* | *97* | *77.69* |
| *f16* | *none* | *117.9* | *83%* | *59* | *47.55* |

Q4_0 does perhaps a third of Q4_K_M's unpacking work and is the *smaller*
file. It is one point **slower** in bandwidth efficiency. All four 4-bit
formats land within 55–61% despite radically different packing. Unpacking cost
is visible only at the extreme — IQ4_NL's lookup table costs about six points —
and it is nowhere near the 22 points that separate 4-bit from f16.

**What the data actually shows** is two different limits:

- At 4-bit the kernel saturates at ~150 G weights/s regardless of format.
- At f16 it saturates at ~118 GB/s, which is 83% of the streaming ceiling.

So f16 is bandwidth-bound and 4-bit is bound by per-weight processing rate. The
crossover sits near 8 bits. Critically, 4-bit saturates **neither** limit:
60% of achievable bandwidth and roughly 304 GFLOP/s, about 8% of this GPU's
fp32 peak. Neither the memory system nor the arithmetic units are busy.

That signature — both resources idle, throughput capped anyway — points to
latency or occupancy in the matrix-vector reduction rather than to any
arithmetic the kernel performs. Each weight is read exactly once and never
reused, so there is no cache reuse to exploit and the kernel must keep enough
loads in flight to cover memory latency. Failing to do so caps throughput
without saturating anything.

**Revised research target.** Not a cheaper quantisation format — that is now
measured and does not help. The opportunity, if there is one, is a
matrix-vector kernel with better latency hiding at low bit-widths on RDNA1
through MoltenVK. That is kernel engineering rather than format design, it is
considerably harder, and this report does not establish how much is available.
The honest statement is that ~40% of achievable bandwidth is unexplained at
4-bit, and neither of the two obvious explanations accounts for it.

**Practical note.** Q4_0 is 4% faster than Q4_K_M at the same bit width and a
slightly smaller file, at some cost in perplexity. That is a real if modest
win for anyone who will trade a little quality for speed.

### 5.1 Efficiency and speed pull in opposite directions

The table above contains an inversion worth stating plainly. The format with
the *best* bandwidth efficiency, f16 at 83%, is the *slowest* at 47.55 tok/s.
The format with the worst efficiency, Q4_0 at 60%, is the fastest at
110.73 tok/s.

Higher bit widths use the memory system better but have more bytes to move.
"Maximise bandwidth efficiency" and "maximise tokens per second" are therefore
different objectives with different answers, and on this hardware they are
close to opposed. Anyone optimising here should decide which one they mean.

## 6. Positive result: n-gram cache drafting

`--spec-type ngram-cache` drafts continuations by matching against text already
in the context, costing no additional weight traffic. Two independent
drift-controlled runs:

| workload | control | ngram-cache | |
|---|---|---|---|
| prose, no repetition | 45.12 | 45.18 | unchanged |
| output echoing input | 40.57 | **48.38** | **+19%** |

The gain appears wherever output repeats input — editing code, refactoring,
summarising a file — which is most of what a local model is used for. It is
enabled by default in this fork for Vulkan devices on darwin.

## 7. Thermal behaviour dominates in practice

None of the above is what a user actually experiences. Under continuous load,
identical requests to a single verified server:

    45.70  45.71  45.05  44.70  44.32  44.08  43.62  43.39  41.39  36.88  32.03  27.78
    └────────── stable ~36 s ─────────┘└─────────────── −39% ───────────────┘

Throughput falls 39% within sixty seconds and had not plateaued when the
measurement ended. On a seven-year-old chassis dissipating a busy discrete GPU,
sustained performance is set by cooling, not by any software property measured
in this report. Peak throughput is a benchmark number; the steady state is the
product.

## 8. Limitations

Single GPU, single machine, single model family. The 141.4 GB/s ceiling is one
kernel's result and a better-tuned streaming kernel might exceed it, which
would lower every "% of achievable" figure proportionally. Thermal state is
controlled by cooldowns and bracketing rather than by measuring die
temperature, which requires elevated privileges. The dequantisation-cost
conclusion is inferred from a monotonic trend across four formats, not from
isolating ALU cost directly — a kernel-level ablation would be stronger
evidence.

## 9. Summary

- Vulkan/MoltenVK gives Intel Macs with AMD GPUs a working GPU inference path:
  **45.7 vs 13.1 tok/s**, 3.5x over CPU.
- The rated 192 GB/s is unreachable; the real ceiling is **141.4 GB/s**.
- Against that ceiling inference already achieves **65–83%**.
- The residual at 4-bit is **not** dequantisation cost: four 4-bit formats with
  very different unpacking all land at 55-61%. It saturates neither bandwidth
  nor arithmetic, and remains unexplained.
- **Efficiency and speed are opposed here**: f16 is most efficient (83%) and
  slowest (47.6 tok/s); Q4_0 is least efficient (60%) and fastest (110.7).
- Native Metal on AMD Macs is **20x worse** than Vulkan through MoltenVK.
- Draft-model speculative decoding cannot pay for itself in this model family;
  **n-gram cache drafting gains 19%** on repetitive workloads.
- **Thermal throttling costs 39% in sixty seconds** and outweighs everything
  else in daily use.

## Reproducing

Scripts are in `vulkan-macos/`. All figures come from a fresh PID-verified
`llama-server` per configuration with warm-up discarded and cooldowns between
runs. `REQUIRE_VULKAN_DEVICE=1 ./vulkan-macos/smoke-test.sh <dist>` asserts the
GPU path is actually exercised rather than silently falling back.
