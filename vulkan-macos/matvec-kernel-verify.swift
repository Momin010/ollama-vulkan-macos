// Correctness check for the R=2 U=4 kernel before believing its throughput.
//
// The benchmark ran on uninitialised private buffers and never inspected the
// output, so a kernel that silently skipped work would have looked fast. This
// fills real q4_0 data, runs both the simple and the unrolled kernel, and
// compares each against a CPU reference.
import Metal
import Foundation

let M = 256          // small enough to check exhaustively on the CPU
let K = 2048
let nblk = K / 32

let src = """
#include <metal_stdlib>
using namespace metal;

template <uint R, uint U>
inline void q4_mlp(device const uint4* q, device const float* sc,
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
            float d = (sc[(ulong)(row0 + r) * nblk + b]);
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

kernel void k_r2u4(device const uint4* q [[buffer(0)]], device const float4* x [[buffer(1)]],
                   device const float* sc [[buffer(2)]], device float* out [[buffer(3)]],
                   constant uint& nblk [[buffer(4)]],
                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]])
{ q4_mlp<2,4>(q, sc, x, out, nblk, (tg * 8 + sg) * 2, lane); }

kernel void k_r1u1(device const uint4* q [[buffer(0)]], device const float4* x [[buffer(1)]],
                   device const float* sc [[buffer(2)]], device float* out [[buffer(3)]],
                   constant uint& nblk [[buffer(4)]],
                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]])
{ q4_mlp<1,1>(q, sc, x, out, nblk, (tg * 8 + sg) * 1, lane); }
"""

setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!
let cq = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: nil)

// deterministic pseudo-random data
var seed: UInt64 = 0x2545F4914F6CDD1D
func rnd() -> Float {
    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
    return Float(seed % 1000) / 1000.0 - 0.5
}

var qdata = [UInt32](repeating: 0, count: M * nblk * 4)
for i in 0..<qdata.count { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; qdata[i] = UInt32(seed & 0xFFFFFFFF) }
var scdata = [Float](repeating: 0, count: M * nblk)
for i in 0..<scdata.count { scdata[i] = rnd() * 0.1 + 0.15 }
var xdata = [Float](repeating: 0, count: K)
for i in 0..<K { xdata[i] = rnd() }

// CPU reference, mirroring the kernel's nibble ordering exactly
func reference(_ row: Int) -> Float {
    var total: Float = 0
    for b in 0..<nblk {
        let d = scdata[row * nblk + b]
        var s: Float = 0, m: Float = 0
        for i in 0..<4 {
            let qi = qdata[(row * nblk + b) * 4 + i]
            for j in 0..<4 {
                let lo = Float((qi >> (8 * UInt32(j))) & 0xF)
                let hi = Float((qi >> (8 * UInt32(j) + 4)) & 0xF)
                s += lo * xdata[b * 32 + i * 8 + j]
                s += hi * xdata[b * 32 + i * 8 + 4 + j]
                m += xdata[b * 32 + i * 8 + j] + xdata[b * 32 + i * 8 + 4 + j]
            }
        }
        total += d * s - 8.0 * d * m
    }
    return total
}

let qb = dev.makeBuffer(bytes: qdata, length: qdata.count * 4, options: .storageModeManaged)!
let sb = dev.makeBuffer(bytes: scdata, length: scdata.count * 4, options: .storageModeManaged)!
let xb = dev.makeBuffer(bytes: xdata, length: xdata.count * 4, options: .storageModeManaged)!
let ob = dev.makeBuffer(length: M * 4, options: .storageModeShared)!

for (name, R) in [("k_r1u1", 1), ("k_r2u4", 2)] {
    memset(ob.contents(), 0, M * 4)
    let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    var n = UInt32(nblk)
    let cb = cq.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(p)
    enc.setBuffer(qb, offset: 0, index: 0); enc.setBuffer(xb, offset: 0, index: 1)
    enc.setBuffer(sb, offset: 0, index: 2); enc.setBuffer(ob, offset: 0, index: 3)
    enc.setBytes(&n, length: 4, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: M / (R * 8), height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 32 * 8, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    let got = ob.contents().bindMemory(to: Float.self, capacity: M)
    var worst = 0.0, nonzero = 0
    for r in 0..<M {
        let want = reference(r)
        if abs(want) > 1e-3 { nonzero += 1 }
        let rel = Double(abs(got[r] - want)) / Double(max(abs(want), 1e-3))
        worst = max(worst, rel)
    }
    print("\(name): worst relative error " + String(format: "%.6f", worst) + "   nonzero rows \(nonzero)/\(M)   sample got=" + String(format: "%.4f", got[7]) + " want=" + String(format: "%.4f", reference(7)))
}
