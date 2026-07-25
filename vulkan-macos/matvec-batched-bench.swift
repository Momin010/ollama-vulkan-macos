// Per-token matvec benchmark, batched the way real inference submits work.
//
// The previous harness called commit() + waitUntilCompleted() per dispatch and
// multiplied by the op count. llama.cpp encodes a whole graph into one command
// buffer and submits once, so the old numbers carried ~197 CPU/GPU round trips
// that real decoding never pays. That is the likely reason the simulation
// predicted 17.4 tok/s where llama.cpp actually delivers 45.68.
//
// This version measures the round-trip cost explicitly, then encodes an entire
// token's matvecs into a single command buffer and times that.
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

kernel void nop(device float* out [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t == 0) out[0] = 1.0;
}

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
                    float4 lo = float4( qi & 0xF, (qi >> 8) & 0xF, (qi >> 16) & 0xF, (qi >> 24) & 0xF);
                    float4 hi = float4((qi >> 4) & 0xF, (qi >> 12) & 0xF, (qi >> 20) & 0xF, (qi >> 28) & 0xF);
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
                float4 lo = float4( qi & 0xF, (qi >> 8) & 0xF, (qi >> 16) & 0xF, (qi >> 24) & 0xF);
                float4 hi = float4((qi >> 4) & 0xF, (qi >> 12) & 0xF, (qi >> 20) & 0xF, (qi >> 28) & 0xF);
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
                float4 lo = float4( qi & 0xF, (qi >> 8) & 0xF, (qi >> 16) & 0xF, (qi >> 24) & 0xF);
                float4 hi = float4((qi >> 4) & 0xF, (qi >> 12) & 0xF, (qi >> 20) & 0xF, (qi >> 28) & 0xF);
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

#define SPLITK(NAME, U) \\
kernel void NAME(device const uint4* q [[buffer(0)]], device const float4* x [[buffer(1)]], \\
                 device const half* sc [[buffer(2)]], device float* out [[buffer(3)]], \\
                 constant uint& nblk [[buffer(4)]], \\
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \\
                 uint lane [[thread_index_in_simdgroup]]) \\
{ split_mv<2,U>(q, sc, x, out, nblk, (tg * 8 + sg) * 2, lane); }

SPLITK(s2, 2)
SPLITK(s3, 3)
SPLITK(s4, 4)

kernel void g2(device const ushort* w [[buffer(0)]], device const float4* x [[buffer(1)]],
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
func pipe(_ n: String) -> MTLComputePipelineState {
    try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!)
}
let pNop = pipe("nop"), pG = pipe("g2")
let splits = [2: pipe("s2"), 3: pipe("s3"), 4: pipe("s4")]

func buf(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: max(n, 256), options: .storageModePrivate)! }

fputs("cooling 150s\n", stderr)
Thread.sleep(forTimeInterval: 150)

// --- 1. cost of one commit/wait round trip -----------------------------------
var rt = Double.greatestFiniteMagnitude
let dummy = buf(1024)
for it in 0..<40 {
    let t0 = Date()
    let cb = cq.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pNop); e.setBuffer(dummy, offset: 0, index: 3)
    e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if it >= 10 { rt = min(rt, Date().timeIntervalSince(t0)) }
}
print(String(format: "\nper-dispatch commit+wait round trip: %.3f ms", rt * 1000))
let totalOps = shapes.reduce(0) { $0 + $1.count }
print(String(format: "with %d matvecs per token that is %.1f ms of pure sync in the old harness\n",
             totalOps, rt * 1000 * Double(totalOps)))

// --- 2. whole token in ONE command buffer -------------------------------------
struct Res { let w: MTLBuffer; let sc: MTLBuffer; let x: MTLBuffer; let out: MTLBuffer; let nblk: Int }
var splitRes: [Res] = [], glueRes: [Res] = []
for sh in shapes {
    let nb = sh.k / 32
    splitRes.append(Res(w: buf(sh.m*nb*16), sc: buf(sh.m*nb*2), x: buf(sh.k*4), out: buf(sh.m*4), nblk: nb))
    glueRes.append(Res(w: buf(sh.m*nb*18), sc: buf(sh.m*nb*2), x: buf(sh.k*4), out: buf(sh.m*4), nblk: nb))
}
// unroll depth chosen per shape, from the previous run
let bestU = [2, 4, 2, 4, 3]

func runToken(split: Bool) -> Double {
    var best = Double.greatestFiniteMagnitude
    for it in 0..<8 {
        let cb = cq.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        for (i, sh) in shapes.enumerated() {
            let r = split ? splitRes[i] : glueRes[i]
            let p = split ? splits[bestU[i]]! : pG
            var n = UInt32(r.nblk)
            enc.setComputePipelineState(p)
            enc.setBuffer(r.w, offset: 0, index: 0); enc.setBuffer(r.x, offset: 0, index: 1)
            enc.setBuffer(r.sc, offset: 0, index: 2); enc.setBuffer(r.out, offset: 0, index: 3)
            enc.setBytes(&n, length: 4, index: 4)
            let groups = max(sh.m / 16, 1)
            for _ in 0..<sh.count {
                enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let dt = cb.gpuEndTime - cb.gpuStartTime
        if it >= 2 && dt > 0 { best = min(best, dt) }
    }
    return best
}

let tg = runToken(split: false)
let ts = runToken(split: true)
let other = 0.00220

print(String(format: "one token, all %d matvecs in a single command buffer:", totalOps))
print(String(format: "  native 18-byte layout : %6.2f ms   -> %5.1f tok/s", tg*1000, 1.0/(tg+other)))
print(String(format: "  split layout          : %6.2f ms   -> %5.1f tok/s", ts*1000, 1.0/(ts+other)))
print(String(format: "  layout speedup        : %.2fx", tg/ts))
print(String(format: "\nllama.cpp measured on this machine: 45.68 tok/s (q4_K_M)"))
