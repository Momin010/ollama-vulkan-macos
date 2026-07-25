# Maintaining this fork

Notes for whoever keeps this working. The user-facing documentation is in
[`.github/README.md`](../.github/README.md).

## The shape of the thing

The fork is a **patch series rebased onto an upstream release tag**. It is
never merged. `vulkan-macos/UPSTREAM_BASE` records which tag the series
currently sits on.

```
v0.32.3 (upstream tag)
  └─ discover: key Vulkan devices by their backend ordinal
     └─ discover: classify Intel and Apple Vulkan GPUs as integrated
        └─ llm: force flash attention off for Vulkan on darwin
           └─ discover: set MoltenVK ICD and f16 defaults
              └─ vulkan-macos: packaging, installer and sync automation
```

Two properties make this maintainable, and both are worth preserving:

**The patches are atomic.** One concern per commit. When upstream changes
something underneath, the conflict lands on one identifiable patch instead of
one large opaque diff.

**The packaging never touches upstream paths.** Everything this fork adds is
under `vulkan-macos/`, `.github/README.md`, or `.github/workflows/vulkan-*`.
Upstream has no files at any of those paths, so packaging can never be what
conflicts. In particular the README lives at `.github/README.md` — GitHub
renders that in preference to the root `README.md`, so the front page is ours
without modifying a file upstream edits constantly.

If you add anything, keep it on a path upstream does not use.

## Updating to a new upstream release

Normally you do nothing: the `vulkan-macos-sync` workflow runs daily, rebases,
verifies on an Intel Mac runner, and publishes. It only publishes if the build,
the unit tests and the packaging smoke test all pass.

By hand:

```sh
git fetch upstream --tags
base=$(cat vulkan-macos/UPSTREAM_BASE)
target=v0.33.0

git rebase --onto "$target" "$base" vulkan-darwin
echo "$target" > vulkan-macos/UPSTREAM_BASE
git commit -am "vulkan-macos: track upstream $target"
git push --force-with-lease

git tag -a "$target-vulkan.1" -m "Ollama $target with Vulkan/MoltenVK"
git push origin "$target-vulkan.1"     # this triggers the release build
```

## When the sync bot opens an issue

It opens one for either of two situations, and they mean different things.

**Rebase conflict.** Upstream edited a line one of the patches touches. Usually
mechanical. The issue names the patch that stopped and the conflicting files.

**Clean rebase, failed verification.** More interesting: the patches still
apply, but upstream changed behaviour around them. Read the upstream diff for
the four files the patches touch before assuming the patch is wrong.

Either way the release branch is left untouched, so the currently published
release stays installable while you sort it out.

## Things that will bite you eventually

**GitHub pauses scheduled workflows after 60 days without repository
activity.** If the sync bot goes quiet for two months, check whether the
schedule was disabled rather than assuming upstream stopped releasing. Pushing
any commit re-enables it.

**Scheduled workflows are disabled by default in forked repositories.** They
were enabled here; if the repo is ever re-forked, this needs doing again.

**Intel macOS runners are on borrowed time.** `macos-13` was already retired;
this uses `macos-15-intel`. When that goes, either cross-compile x86_64 from an
arm64 runner (the Vulkan SDK ships universal dylibs, so it is feasible) or
build locally and upload to the release page by hand. Nothing else in the
pipeline depends on the runner architecture.

**Ollama's auto-update overwrites the patched binary** on users' machines with
no error — inference silently returns to the CPU. The installer's watchdog
handles this, but anyone who installed with `--no-watchdog` will just
experience it as "it got slow again".

## Hardware coverage

Everything has been verified on exactly one machine: a 2019 16" MacBook Pro,
AMD Radeon Pro 5500M 8 GB, alongside an Intel UHD 630. Nothing in the patches
is specific to that GPU, but no other card has been confirmed.

The two behaviours most likely to differ on other hardware:

- `GGML_VK_DISABLE_F16=1` works around a MoltenVK fp16 fault. A newer AMD card
  might not need it, and would give up some performance for nothing. It is a
  default, not a forced value, so `GGML_VK_DISABLE_F16=0` tests that.
- The integrated-GPU classifier matches on vendor strings because MoltenVK
  exposes no vendor ID and leaves `PCIID` empty on this path. A machine
  reporting unusual device names could be misclassified. `OLLAMA_IGPU_ENABLE=1`
  is the escape hatch.

## Releasing without CI

If Actions is unavailable, the release artifact can be built locally:

```sh
brew install cmake molten-vk vulkan-loader vulkan-headers glslang shaderc
export VULKAN_SDK=$(brew --prefix)

cmake -B build -DOLLAMA_LLAMA_BACKENDS=vulkan
cmake --build build --parallel 8
go build .

./vulkan-macos/package.sh build dist/ollama-vulkan-macos
./vulkan-macos/smoke-test.sh dist/ollama-vulkan-macos

tar -czf ollama-vulkan-macos-<tag>-x86_64.tar.gz -C dist ollama-vulkan-macos
shasum -a 256 ollama-vulkan-macos-<tag>-x86_64.tar.gz > checksums.txt
```

Attach both files to a GitHub release. The installer looks for an asset matching
`ollama-vulkan-macos-*.tar.gz` and, if present, verifies it against
`checksums.txt` — so upload the checksum file too, or installs will warn.
