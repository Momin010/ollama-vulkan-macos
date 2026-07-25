# Ollama with GPU acceleration for Intel Macs

**Run Ollama on the AMD GPU in your Intel MacBook Pro instead of the CPU.**

Official Ollama uses Metal for GPU acceleration, and Metal support only covers
Apple Silicon. On an Intel Mac with a discrete AMD card, Ollama quietly falls
back to the CPU — the GPU sits idle while your laptop gets hot and slow.

This is a fork of [ollama/ollama](https://github.com/ollama/ollama) that routes
inference through **Vulkan** via **MoltenVK**, which does support these GPUs.

Measured on a 2019 16" MacBook Pro, AMD Radeon Pro 5500M 8 GB:

| Model | Official Ollama (CPU) | This build (GPU) | |
|---|---|---|---|
| Llama 3.2 3B Q4_K_M | 14 tok/s | **36 tok/s** | 2.6× |
| Qwen 2.5 Coder 7B Q4_K_M | 5.7 tok/s | **22 tok/s** | 3.9× |

For reference, a hand-built llama.cpp with Vulkan reaches 38 tok/s on the 3B —
so this is within about 5% of the practical ceiling for this hardware.

## Install

You need Ollama already installed. This patches it in place.

```sh
curl -fsSL https://raw.githubusercontent.com/Momin010/ollama-vulkan-macos/vulkan-darwin/vulkan-macos/install.sh | bash
```

That's it. Everything else about Ollama works exactly as before — same CLI,
same API on port 11434, same models, same desktop app.

The download is self-contained: MoltenVK and the Vulkan loader are bundled, so
you do **not** need Homebrew, the Vulkan SDK, Xcode, or Go.

### Reverting

The installer keeps a copy of the official binary and can put it back:

```sh
curl -fsSL https://raw.githubusercontent.com/Momin010/ollama-vulkan-macos/vulkan-darwin/vulkan-macos/install.sh | bash -s -- --uninstall
```

## Does this apply to me?

**Yes** if you have an Intel Mac with a discrete AMD GPU — the 2016–2019
MacBook Pro 15"/16", iMac, and iMac Pro lines, plus Macs with an eGPU.

**No** if you have an Apple Silicon Mac (M1 and later). Those already run on
the GPU through Metal, which is faster than this path. The installer detects
Apple Silicon and refuses to run.

**Probably not** if your Intel Mac has only Intel integrated graphics. The
integrated GPU is slower than the CPU for this workload, so this build
deliberately ignores it.

## Checking it worked

```sh
# which device is actually being used
grep 'inference compute' ~/.ollama/logs/server.log | tail -1

# generation speed
ollama run llama3.2 --verbose
```

The log line should say `library=Vulkan` and name your AMD card. In
`ollama run --verbose` output, the number that matters is `eval rate`.

## Known limitations

**Ollama's auto-update will silently undo this.** If the desktop app updates
itself, it replaces the patched binary with the official one and you are back
on CPU with no error message — just slower responses. Re-running the installer
fixes it. Declining update prompts avoids it.

**Flash attention is permanently disabled** on this path. Under MoltenVK,
llama.cpp assigns the flash-attention tensor to the CPU, which forces a round
trip per token and drops generation to under 1 tok/s. The build forces it off
and ignores `OLLAMA_FLASH_ATTENTION`.

**fp16 matmul is disabled** (`GGML_VK_DISABLE_F16=1` by default). MoltenVK's
fp16 path crashes the GPU with `vk::ErrorDeviceLost` on prompts over roughly
1–2k tokens. The f32 path is stable and, on this hardware, actually faster at
prompt processing.

**8 GB of VRAM is the real constraint.** A 7B model at Q4 plus its KV cache
fits with a 4096–8192 context. Larger contexts or two models loaded at once
will spill layers to the CPU and get much slower. `ollama ps` shows the split.

**Only tested on the Radeon Pro 5500M.** Other AMD cards should work — nothing
in the changes is specific to that chip — but nobody has confirmed it. Reports
welcome.

## What was actually changed

Four patches on top of upstream, each in its own commit:

| Change | Why |
|---|---|
| Key Vulkan devices by backend ordinal, not list position | macOS lists an Accelerate BLAS pseudo-device first, shifting every Vulkan device ID by one. Selecting "GPU 0" actually ran on the Intel iGPU, producing garbage output at a sixth of the speed. |
| Classify Intel/Apple Vulkan GPUs as integrated on darwin | MoltenVK emits no `uma` metadata, so the iGPU arrived unclassified reporting system RAM as VRAM — inflating the auto-selected context to 32768 and competing for device selection. |
| Force flash attention off for Vulkan on darwin | See above; 35 tok/s → 0.7 tok/s otherwise. |
| Set MoltenVK ICD and f16 defaults | The desktop app launches the server with a minimal environment, so these cannot be left to the user's shell profile. |

No changes to CMake were needed; the Vulkan backend already builds on macOS.

## Staying current with upstream

The patches are maintained as a **rebase onto an upstream release tag**, never
a merge. A scheduled workflow checks daily for a new stable Ollama release,
replays the patch series onto it, and — only if it builds, passes tests and
passes a packaging smoke test on a real Intel Mac runner — publishes a new
release automatically. If anything fails, it opens an issue instead and leaves
the current release untouched.

This works because the patches touch quiet parts of the tree. Across the 73
commits from v0.31.1 to v0.32.3, the three files they modify changed 0, 1 and 2
times respectively, and the series still applied without conflict.

The upstream release this fork currently tracks is recorded in
[`vulkan-macos/UPSTREAM_BASE`](../vulkan-macos/UPSTREAM_BASE).

## Building it yourself

```sh
brew install cmake molten-vk vulkan-loader vulkan-headers glslang shaderc
export VULKAN_SDK=$(brew --prefix)

cmake -B build -DOLLAMA_LLAMA_BACKENDS=vulkan
cmake --build build --parallel 8
go build .

./vulkan-macos/package.sh build dist/ollama-vulkan-macos
./vulkan-macos/smoke-test.sh dist/ollama-vulkan-macos
```

## Credit and licence

All of the hard work here is [Ollama's](https://github.com/ollama/ollama) and
[llama.cpp's](https://github.com/ggml-org/llama.cpp); this fork is four small
patches and some packaging. MIT licensed, same as upstream.

This is an unofficial community build. Please do not report issues with it to
the Ollama project — open them
[here](https://github.com/Momin010/ollama-vulkan-macos/issues) instead.
