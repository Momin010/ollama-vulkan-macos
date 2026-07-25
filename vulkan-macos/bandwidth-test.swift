// Minimal streaming-read bandwidth benchmark for a Metal GPU.
//
// The LLM workload is dominated by reading weights once per token, so the
// relevant hardware number is not the datasheet bus rate but how fast this GPU
// can actually stream a large buffer. Each thread reads a strided run of
// float4s and accumulates, so the kernel is pure sequential read with a
// trivial ALU tail -- an upper bound on what any inference kernel could reach.
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;

kernel void stream_read(device const float4* src   [[buffer(0)]],
                        device float*        out   [[buffer(1)]],
                        constant uint&       n4    [[buffer(2)]],
                        uint tid  [[thread_position_in_grid]],
                        uint nthr [[threads_per_grid]])
{
    float4 acc = float4(0.0);
    for (uint i = tid; i < n4; i += nthr) {
        acc += src[i];
    }
    // Keep the accumulator alive so nothing is optimised away.
    out[tid] = acc.x + acc.y + acc.z + acc.w;
}
"""

guard let dev = MTLCreateSystemDefaultDevice() else {
    print("no Metal device"); exit(1)
}
print("device: \(dev.name)")
print("unified memory: \(dev.hasUnifiedMemory)")
print(String(format: "recommendedMaxWorkingSetSize: %.0f MB",
             Double(dev.recommendedMaxWorkingSetSize) / 1e6))

let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) }
catch { print("shader compile failed: \(error)"); exit(1) }

guard let fn = lib.makeFunction(name: "stream_read") else { print("no fn"); exit(1) }
let pipe: MTLComputePipelineState
do { pipe = try dev.makeComputePipelineState(function: fn) }
catch { print("pipeline failed: \(error)"); exit(1) }

guard let q = dev.makeCommandQueue() else { print("no queue"); exit(1) }

// 1 GiB buffer, private storage so it lives in VRAM.
let bytes = 1 << 30
let n4 = UInt32(bytes / 16)
guard let buf = dev.makeBuffer(length: bytes, options: .storageModePrivate) else {
    print("alloc failed"); exit(1)
}

let nthreads = 1 << 20
guard let out = dev.makeBuffer(length: nthreads * 4, options: .storageModePrivate) else {
    print("out alloc failed"); exit(1)
}

var n4v = n4
var best = 0.0
for run in 1...7 {
    let t0 = Date()
    guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { break }
    enc.setComputePipelineState(pipe)
    enc.setBuffer(buf, offset: 0, index: 0)
    enc.setBuffer(out, offset: 0, index: 1)
    enc.setBytes(&n4v, length: 4, index: 2)
    let tg = min(pipe.maxTotalThreadsPerThreadgroup, 256)
    enc.dispatchThreads(MTLSize(width: nthreads, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let dt = Date().timeIntervalSince(t0)
    let gbs = Double(bytes) / dt / 1e9
    if run > 2 { best = max(best, gbs) }   // discard warm-up
    print(String(format: "  run %d: %6.1f GB/s  (%.1f ms)", run, gbs, dt * 1000))
}

print(String(format: "\nBEST STREAMING READ: %.1f GB/s", best))
print(String(format: "theoretical bus peak: 192.0 GB/s  ->  %.0f%% of theoretical", 100 * best / 192.0))
print(String(format: "LLM best measured (1B f16): 117.9 GB/s  ->  %.0f%% of this achievable ceiling",
             100 * 117.9 / max(best, 1)))
