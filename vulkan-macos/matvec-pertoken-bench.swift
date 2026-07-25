// Simulate one token's worth of matrix-vector work for Llama 3.2 3B, using the
// real per-layer shapes taken from a GGML_VK_PERF_LOGGER profile, and time it
// under both weight layouts.
//
// The 47% figure so far comes from a single 8192x8192 matrix. Real decoding is
// dozens of differently shaped matvecs, and the profile showed small ones
// running far below the large ones (56 GB/s for a 74 MB op versus 103 GB/s for
// a 323 MB one). If the layout advantage collapses on small shapes, the
// end-to-end gain will be much less than the microbenchmark suggests.
//
// Shapes per token, from the profile:
//     m=3072  k=3072   x56    attention q/k/v/o projections
//     m=8192  k=3072   x56    ffn gate + up
//     m=3072  k=8192   x28    ffn down
//     m=1024  k=3072   x56    kv projections
//     m=128256 k=3072  x1     output head over the vocabulary
import Metal
import Foundation

struct Shape { let m: Int; let k: Int; let count: Int; let label: String }
let shapes = [
    Shape(m: 3072,   k: 3072, count: 56, label: "attn qkv/o"),
    Shape(m: 8192,   k: 3072, count: 56, label: "ffn gate+up"),
    Shape(m: 3072,   k: 8192, count: 28, label: "ffn down"),
    Shape(m: 1024,   k: 3072, count: 56, label: "kv proj"),
    Shape(m: 128256, k: 3072, count: 1,  label: "output head"),
]

let src = """
#include <metal_stdlib>
using namespace metal;

template <uint R, uint U>
inline void split_mv(device const uint4* q, device const half* sc,
                     device const float4* x, device float* out,
                     uint nblk, uint row0, uint lane)
{
    float acc[R], accm[R];
    for (uint r = 0; r < R; ++r) { acc[r] = 0.0; accm[r] = 0.0; }
    uint b = lane;
    for (; b + (U - 1) * 32 < nblk; b += U * 32) {
        uint4 qv[R][U]; half dv[R][U];
        for (uint u = 0; u < U; ++u)
            for (uint r = 0; r < R; ++r) {
                qv[r][u] = q[(ulong)(row0 + r) * nblk + b + u * 32];
                dv[r][u] = sc[(ulong)(row0 + r) * nblk + b + u * 32];
            }
        for (uint u = 0; u < U; ++u) {
            device const float4* xv = x + (b + u * 32) * 8;
            float4 a[8];
            for (uint i = 0; i < 8; ++i) a[i] = xv[i];
            for (uint r = 0; r < R; ++r) {
                float s = 0.0, m = 0.0;
                for (uint i = 0; i < 4; ++i) {
                    uint qi = qv[r][u][i];
                    float4 lo = float4( qi        & 0xF, (qi >>  8) & 0xF,
                                       (qi >> 16) & 0xF, (qi >> 24) & 0xF);
                    float4 hi = float4((qi >>  4) & 0xF, (qi >> 12) & 0xF,
                                       (qi >> 20) & 0xF, (qi >> 28) & 0xF);
                    s += dot(lo, a[i*2]) + dot(hi, a[i*2+1]);
                    m += dot(float4(1), a[i*2]) + dot(float4(1), a[i*2+1]);
                }
                acc[r] += float(dv[r][u]) * s;  accm[r] += float(dv[r][u]) * m;
            }
        }
    }
    for (; b < nblk; b += 32) {
        device const float4* xv = x + b * 8;
        float4 a[8];
        for (uint i = 0; i < 8; ++i) a[i] = xv[i];
        for (uint r = 0; r < R; ++r) {
            uint4 qv1 = q[(ulong)(row0 + r) * nblk + b];
            float d = float(sc[(ulong)(row0 + r) * nblk + b]);
            float s = 0.0, m = 0.0;
            for (uint i = 0; i < 4; ++i) {
                uint qi = qv1[i];
                float4 lo = float4( qi        & 0xF, (qi >>  8) & 0xF,
                                   (qi >> 16) & 0xF, (qi >> 24) & 0xF);
                float4 hi = float4((qi >>  4) & 0xF, (qi >> 12) & 0xF,
                                   (qi >> 20) & 0xF, (qi >> 28) & 0xF);
                s += dot(lo, a[i*2]) + dot(hi, a[i*2+1]);
                m += dot(float4(1), a[i*2]) + dot(float4(1), a[i*2+1]);
            }
            acc[r] += d * s;  accm[r] += d * m;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float v = simd_sum(acc[r] - 8.0 * accm[r]);
        if (lane == 0) out[row0 + r] = v;
    }
}

// Native GGUF layout, 18-byte blocks read as ushorts.
template <uint R>
inline void glue_mv(device const ushort* w, device const float4* x,
                    device float* out, uint nblk, uint row0, uint lane)
{
    float acc[R], accm[R];
    for (uint r = 0; r < R; ++r) { acc[r] = 0.0; accm[r] = 0.0; }
    for (uint b = lane; b < nblk; b += 32) {
        device const float4* xv = x + b * 8;
        float4 a[8];
        for (uint i = 0; i < 8; ++i) a[i] = xv[i];
        for (uint r = 0; r < R; ++r) {
            ulong base = ((ulong)(row0 + r) * nblk + b) * 9;
            float d = float(as_type<half>(w[base]));
            float s = 0.0, m = 0.0;
            for (uint i = 0; i < 4; ++i) {
                uint qi = (uint)w[base + 1 + i*2] | ((uint)w[base + 2 + i*2] << 16);
                float4 lo = float4( qi        & 0xF, (qi >>  8) & 0xF,
                                   (qi >> 16) & 0xF, (qi >> 24) & 0xF);
                float4 hi = float4((qi >>  4) & 0xF, (qi >> 12) & 0xF,
                                   (qi >> 20) & 0xF, (qi >> 28) & 0xF);
                s += dot(lo, a[i*2]) + dot(hi, a[i*2+1]);
                m += dot(float4(1), a[i*2]) + dot(float4(1), a[i*2+1]);
            }
            acc[r] += d * s;  accm[r] += d * m;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float v = simd_sum(acc[r] - 8.0 * accm[r]);
        if (lane == 0) out[row0 + r] = v;
    }
}

kernel void k_split(device const uint4* q [[buffer(0)]], device const float4* x [[buffer(1)]],
                    device const half* sc [[buffer(2)]], device float* out [[buffer(3)]],
                    constant uint& nblk [[buffer(4)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]])
{ split_mv<2,4>(q, sc, x, out, nblk, (tg * 8 + sg) * 2, lane); }

kernel void k_split3(device const uint4* q [[buffer(0)]], device const float4* x [[buffer(1)]],
                     device const half* sc [[buffer(2)]], device float* out [[buffer(3)]],
                     constant uint& nblk [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]])
{ split_mv<2,3>(q, sc, x, out, nblk, (tg * 8 + sg) * 2, lane); }

kernel void k_split2(device const uint4* q [[buffer(0)]], device const float4* x [[buffer(1)]],
                     device const half* sc [[buffer(2)]], device float* out [[buffer(3)]],
                     constant uint& nblk [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]])
{ split_mv<2,2>(q, sc, x, out, nblk, (tg * 8 + sg) * 2, lane); }

kernel void k_glue(device const ushort* w [[buffer(0)]], device const float4* x [[buffer(1)]],
                   device const half* sc [[buffer(2)]], device float* out [[buffer(3)]],
                   constant uint& nblk [[buffer(4)]],
                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]])
{ glue_mv<2>(w, x, out, nblk, (tg * 8 + sg) * 2, lane); }
"""

setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!
let cq = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let pSplit  = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "k_split")!)
let pSplit3 = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "k_split3")!)
let pSplit2 = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "k_split2")!)
let pGlue  = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "k_glue")!)

func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 256), options: .storageModePrivate)! }

func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }
func lpad(_ s: String, _ n: Int) -> String { s.count >= n ? s : String(repeating: " ", count: n - s.count) + s }

fputs("cooling 150s\n", stderr)
Thread.sleep(forTimeInterval: 150)

// Time one instance of a shape, return best seconds over several dispatches.
func time(_ pipe: MTLComputePipelineState, _ w: MTLBuffer, _ x: MTLBuffer,
          _ sc: MTLBuffer, _ out: MTLBuffer, m: Int, nblk: Int) -> Double {
    var n = UInt32(nblk)
    var best = Double.greatestFiniteMagnitude
    let groups = max(m / (2 * 8), 1)
    for it in 0..<12 {
        let cb = cq.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(w, offset: 0, index: 0); enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(sc, offset: 0, index: 2); enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32 * 8, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let dt = cb.gpuEndTime - cb.gpuStartTime
        if it >= 3 && dt > 0 { best = min(best, dt) }
    }
    return best
}

print("\nper-token matvec workload, Llama 3.2 3B shapes\n")
print(pad("shape", 22) + lpad("count", 7) + lpad("MB", 9)
      + lpad("glue ms", 11) + lpad("split ms", 11) + lpad("speedup", 10) + lpad("best", 7))
print(String(repeating: "-", count: 70))

var totGlue = 0.0, totSplit = 0.0, totMB = 0.0
for sh in shapes {
    let nblk = sh.k / 32
    let wsplit = buf(sh.m * nblk * 16)
    let wglue  = buf(sh.m * nblk * 18)
    let sc     = buf(sh.m * nblk * 2)
    let x      = buf(sh.k * 4)
    let out    = buf(sh.m * 4)

    let tg = time(pGlue,  wglue,  x, sc, out, m: sh.m, nblk: nblk) * Double(sh.count)
    var ts = Double.greatestFiniteMagnitude
    var bestU = 0
    for (pipe, u) in [(pSplit, 4), (pSplit3, 3), (pSplit2, 2)] {
        let t = time(pipe, wsplit, x, sc, out, m: sh.m, nblk: nblk) * Double(sh.count)
        if t < ts { ts = t; bestU = u }
    }
    let mb = Double(sh.m) * Double(sh.k) * 0.5625 * Double(sh.count) / 1e6
    totGlue += tg; totSplit += ts; totMB += mb

    print(pad("\(sh.label) \(sh.m)x\(sh.k)", 22) + lpad("\(sh.count)", 7)
          + lpad(String(format: "%.0f", mb), 9)
          + lpad(String(format: "%.2f", tg * 1000), 11)
          + lpad(String(format: "%.2f", ts * 1000), 11)
          + lpad(String(format: "%.2fx", tg / ts), 10) + lpad("U=\(bestU)", 7))
}

print(String(repeating: "-", count: 70))
print(pad("TOTAL matvec", 22) + lpad("", 7) + lpad(String(format: "%.0f", totMB), 9)
      + lpad(String(format: "%.2f", totGlue * 1000), 11)
      + lpad(String(format: "%.2f", totSplit * 1000), 11)
      + lpad(String(format: "%.2fx", totGlue / totSplit), 10))

// The profile measured 2.20 ms per token in non-matmul operations.
let other = 0.00220
print("\nnon-matmul ops (from profile): \(String(format: "%.2f", other * 1000)) ms")
print(String(format: "predicted tok/s  native layout : %.1f", 1.0 / (totGlue + other)))
print(String(format: "predicted tok/s  split layout  : %.1f", 1.0 / (totSplit + other)))
print(String(format: "\nmeasured llama.cpp today: 45.68 tok/s (q4_K_M)"))
