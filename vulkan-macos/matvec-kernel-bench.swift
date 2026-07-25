// Matrix-vector microbenchmark, fourth attempt: memory-level parallelism.
//
// v3 matched ggml (150 vs 152 G weights/s) with 2 rows per simdgroup. The
// sweep showed R=2 beating R=1 and R=4/R=8 regressing, which looks less like
// activation reuse -- the activation vector is small enough to sit in cache --
// and more like loads-in-flight: R=2 issues two independent weight loads per
// iteration, R=8 runs out of registers.
//
// If the limit is memory-level parallelism, the lever is issuing more
// independent loads per thread before consuming any of them. This version
// hoists U blocks' worth of loads to the top of the loop body, so a thread has
// R*U weight loads outstanding at once, and sweeps both dimensions.
import Metal
import Foundation

let M = 8192
let K = 8192
let ITERS = 20
let SIMD = 32
let SGS_PER_TG = 8

let src = """
#include <metal_stdlib>
using namespace metal;

// R rows per simdgroup, U blocks unrolled per iteration.
// All R*U weight loads are issued before any unpacking happens, so they
// overlap in the memory pipeline instead of serialising on each other.
template <uint R, uint U>
inline void q4_mlp(device const uint4*  q,
                   device const half*   sc,
                   device const float4* x,
                   device float*        out,
                   uint nblk, uint row0, uint lane)
{
    float acc[R], accm[R];
    for (uint r = 0; r < R; ++r) { acc[r] = 0.0; accm[r] = 0.0; }

    uint b = lane;
    for (; b + (U - 1) * 32 < nblk; b += U * 32) {
        uint4 qv[R][U];
        half  dv[R][U];
        // Issue every load first. Nothing here depends on anything above it.
        for (uint u = 0; u < U; ++u) {
            for (uint r = 0; r < R; ++r) {
                qv[r][u] = q[(ulong)(row0 + r) * nblk + b + u * 32];
                dv[r][u] = sc[(ulong)(row0 + r) * nblk + b + u * 32];
            }
        }
        // Now consume them.
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
    // tail
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

#define GEN(NAME, R, U)                                                       \\
kernel void NAME(device const uint4*  q  [[buffer(0)]],                       \\
                 device const float4* x  [[buffer(1)]],                       \\
                 device const half*   sc [[buffer(2)]],                       \\
                 device float* out       [[buffer(3)]],                       \\
                 constant uint& nblk     [[buffer(4)]],                       \\
                 uint tg   [[threadgroup_position_in_grid]],                  \\
                 uint sg   [[simdgroup_index_in_threadgroup]],                \\
                 uint lane [[thread_index_in_simdgroup]])                     \\
{                                                                             \\
    q4_mlp<R, U>(q, sc, x, out, nblk, (tg * 8 + sg) * R, lane);               \\
}

GEN(k_r2u3, 2, 3)
GEN(k_r2u4, 2, 4)
GEN(k_r2u5, 2, 5)
GEN(k_r2u6, 2, 6)
GEN(k_r2u8, 2, 8)
GEN(k_r3u4, 3, 4)
GEN(k_r4u4, 4, 4)
GEN(k_r1u4, 1, 4)
"""

setvbuf(stdout, nil, _IONBF, 0)
guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
    print("no Metal device"); exit(1)
}
print("device: \(dev.name)")
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) }
catch { print("MSL compile failed: \(error)"); exit(1) }

func buf(_ n: Int) -> MTLBuffer {
    guard let b = dev.makeBuffer(length: max(n, 256), options: .storageModePrivate) else { print("alloc failed"); exit(1) }
    return b
}
let nblk = K / 32
let wBytes = M * nblk * 16 + M * nblk * 2
let qb = buf(M * nblk * 16), sc = buf(M * nblk * 2), xb = buf(K * 4), ob = buf(M * 4)

func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }
func lpad(_ s: String, _ n: Int) -> String { s.count >= n ? s : String(repeating: " ", count: n - s.count) + s }

print("matrix \(M) x \(K), weights \(String(format: "%.1f", Double(wBytes)/1e6)) MB\n")
print(pad("config", 20) + lpad("loads/iter", 12) + lpad("ms", 9)
      + lpad("wt GB/s", 10) + lpad("G wts/s", 10) + lpad("%of163.8", 10))
print(String(repeating: "-", count: 71))

let configs = [("k_r1u4",1,4), ("k_r2u3",2,3), ("k_r2u4",2,4), ("k_r2u5",2,5),
               ("k_r2u6",2,6), ("k_r2u8",2,8), ("k_r3u4",3,4), ("k_r4u4",4,4)]
var bestOverall = 0.0; var bestName = ""
for (name, R, U) in configs {
    guard let f = lib.makeFunction(name: name),
          let p = try? dev.makeComputePipelineState(function: f) else {
        print(pad("R=\(R) U=\(U)", 20) + "  pipeline failed"); continue
    }
    var n = UInt32(nblk)
    let groups = M / (R * SGS_PER_TG)
    var best = Double.greatestFiniteMagnitude
    for it in 0..<ITERS {
        guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { break }
        enc.setComputePipelineState(p)
        enc.setBuffer(qb, offset: 0, index: 0); enc.setBuffer(xb, offset: 0, index: 1)
        enc.setBuffer(sc, offset: 0, index: 2); enc.setBuffer(ob, offset: 0, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: SIMD * SGS_PER_TG, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let dt = cb.gpuEndTime - cb.gpuStartTime
        if it >= 4 && dt > 0 { best = min(best, dt) }
    }
    let wtGB = Double(wBytes) / best / 1e9
    let gw = Double(M) * Double(K) / best / 1e9
    if wtGB > bestOverall { bestOverall = wtGB; bestName = "R=\(R) U=\(U)" }
    print(pad("R=\(R) U=\(U)", 20) + lpad("\(R*U)", 12)
          + lpad(String(format: "%.2f", best * 1000), 9)
          + lpad(String(format: "%.1f", wtGB), 10)
          + lpad(String(format: "%.0f", gw), 10)
          + lpad(String(format: "%.0f%%", 100 * wtGB / 163.8), 10))
}
print("\nbest: \(bestName) at \(String(format: "%.1f", bestOverall)) GB/s")
print("ggml: 85.4 GB/s / 152 G wts/s      ceiling: 163.8 GB/s")
