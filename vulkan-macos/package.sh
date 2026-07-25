#!/usr/bin/env bash
#
# Assemble a self-contained Ollama + Vulkan payload for Intel macOS.
#
# The build links libggml-vulkan.so against the Vulkan loader wherever it was
# found at build time (typically a Homebrew prefix). Shipping that as-is would
# require every user to install the identical Homebrew packages. This script
# copies the loader and MoltenVK into the payload, rewrites the link paths to
# be relative to the payload itself, and writes a matching ICD manifest, so the
# result runs on a machine with no Homebrew and no Vulkan SDK.
#
# Usage: vulkan-macos/package.sh <build-dir> <staging-dir>
#
#   build-dir    the cmake build directory (contains lib/ollama and the
#                ollama binary is expected at the repository root)
#   staging-dir  created/overwritten; receives the finished payload
#
set -euo pipefail

BUILD_DIR="${1:-build}"
STAGE_DIR="${2:-dist/ollama-vulkan-macos}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "$BUILD_DIR/lib/ollama" ] || die "no payload at $BUILD_DIR/lib/ollama -- run the cmake build first"
[ -f "./ollama" ] || die "no ollama binary at repository root -- run 'go build .' first"

VULKAN_BACKEND="$BUILD_DIR/lib/ollama/vulkan/libggml-vulkan.so"
[ -f "$VULKAN_BACKEND" ] || die "Vulkan backend missing at $VULKAN_BACKEND -- was the build configured with -DOLLAMA_LLAMA_BACKENDS=vulkan?"

# ---------------------------------------------------------------------------
# Locate the loader and MoltenVK to bundle.
#
# Prefer whatever libggml-vulkan.so was actually linked against, so the bundled
# loader is the one the backend was compiled for rather than a same-named
# library that happens to be installed.
# ---------------------------------------------------------------------------
find_linked_vulkan_loader() {
    otool -L "$VULKAN_BACKEND" \
        | awk '/libvulkan\.1\.dylib/ {print $1; exit}'
}

find_moltenvk() {
    local candidate
    for candidate in \
        "${VULKAN_SDK:-}/lib/libMoltenVK.dylib" \
        /usr/local/opt/molten-vk/lib/libMoltenVK.dylib \
        /opt/homebrew/opt/molten-vk/lib/libMoltenVK.dylib \
        /usr/local/lib/libMoltenVK.dylib
    do
        [ -f "$candidate" ] && { printf '%s' "$candidate"; return 0; }
    done
    return 1
}

LOADER_SRC="$(find_linked_vulkan_loader)"
[ -n "$LOADER_SRC" ] || die "could not determine which Vulkan loader $VULKAN_BACKEND links against"
[ -f "$LOADER_SRC" ] || die "linked Vulkan loader not found on disk: $LOADER_SRC"

MOLTENVK_SRC="$(find_moltenvk)" || die "libMoltenVK.dylib not found -- install it with 'brew install molten-vk'"

log "loader:   $LOADER_SRC"
log "moltenvk: $MOLTENVK_SRC"

# ---------------------------------------------------------------------------
# Stage the payload.
# ---------------------------------------------------------------------------
log "staging payload into $STAGE_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/lib/ollama"

cp -R "$BUILD_DIR/lib/ollama/." "$STAGE_DIR/lib/ollama/"
cp "./ollama" "$STAGE_DIR/ollama"
chmod +x "$STAGE_DIR/ollama"

# The MIT licence requires the copyright notices to travel with binary
# distributions, not just with source. This archive is how almost everyone
# receives the software, so the notices ship inside it.
for legal in LICENSE NOTICE; do
    [ -f "$legal" ] || die "$legal is missing from the repository root"
    cp "$legal" "$STAGE_DIR/$legal"
done

VULKAN_DIR="$STAGE_DIR/lib/ollama/vulkan"
[ -d "$VULKAN_DIR" ] || die "staged payload has no vulkan/ directory"

cp "$LOADER_SRC" "$VULKAN_DIR/libvulkan.1.dylib"
cp "$MOLTENVK_SRC" "$VULKAN_DIR/libMoltenVK.dylib"
chmod u+w "$VULKAN_DIR/libvulkan.1.dylib" "$VULKAN_DIR/libMoltenVK.dylib"

# ---------------------------------------------------------------------------
# Rewrite install names so nothing points outside the payload.
#
# The ICD manifest resolves library_path relative to its own location, so
# MoltenVK does not need an install-name rewrite for the loader to find it --
# but its own id is rewritten anyway so that anything linking it directly
# resolves within the payload too.
# ---------------------------------------------------------------------------
log "rewriting install names"
install_name_tool -id "@loader_path/libvulkan.1.dylib" "$VULKAN_DIR/libvulkan.1.dylib"
install_name_tool -id "@loader_path/libMoltenVK.dylib" "$VULKAN_DIR/libMoltenVK.dylib"
install_name_tool -change "$LOADER_SRC" "@loader_path/libvulkan.1.dylib" "$VULKAN_DIR/libggml-vulkan.so"

# ---------------------------------------------------------------------------
# Ship an ICD manifest pointing at the bundled MoltenVK.
# ---------------------------------------------------------------------------
log "writing ICD manifest"
MOLTENVK_API_VERSION="$(
    /usr/bin/python3 - "$MOLTENVK_SRC" <<'PY' 2>/dev/null || echo "1.2.0"
import json, os, sys
lib = sys.argv[1]
# The Homebrew/SDK manifest sits a few levels up from lib/; reuse its
# api_version when we can find it so we do not misreport the driver.
for rel in ("../etc/vulkan/icd.d/MoltenVK_icd.json", "../share/vulkan/icd.d/MoltenVK_icd.json"):
    path = os.path.normpath(os.path.join(os.path.dirname(lib), rel))
    if os.path.exists(path):
        with open(path) as fh:
            print(json.load(fh)["ICD"]["api_version"])
        break
else:
    print("1.2.0")
PY
)"

cat > "$VULKAN_DIR/MoltenVK_icd.json" <<EOF
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "./libMoltenVK.dylib",
        "api_version": "$MOLTENVK_API_VERSION",
        "is_portability_driver": true
    }
}
EOF

# ---------------------------------------------------------------------------
# Re-sign. Editing a Mach-O with install_name_tool invalidates its signature,
# and macOS refuses to load a dylib whose signature does not match its
# contents. Ad-hoc signatures are sufficient for locally installed code.
# ---------------------------------------------------------------------------
log "re-signing modified binaries"
codesign --force --sign - "$VULKAN_DIR/libvulkan.1.dylib"
codesign --force --sign - "$VULKAN_DIR/libMoltenVK.dylib"
codesign --force --sign - "$VULKAN_DIR/libggml-vulkan.so"
codesign --force --sign - "$STAGE_DIR/ollama"

# ---------------------------------------------------------------------------
# Verify the payload is genuinely self-contained before we ship it. This is the
# check that catches the failure users would otherwise hit as "works on the
# build machine, broken everywhere else".
# ---------------------------------------------------------------------------
log "verifying payload has no external dependencies"
leaked=0
while read -r dep; do
    case "$dep" in
        /usr/lib/*|/System/*|@loader_path/*|@rpath/*|@executable_path/*) ;;
        *) warn "external dependency: $dep"; leaked=1 ;;
    esac
done < <(otool -L "$VULKAN_DIR/libggml-vulkan.so" "$VULKAN_DIR/libvulkan.1.dylib" "$VULKAN_DIR/libMoltenVK.dylib" \
            | awk '/^\t/ {print $1}' | sort -u)

[ "$leaked" -eq 0 ] || die "payload still references libraries outside the bundle (see warnings above)"

log "payload ready: $STAGE_DIR"
du -sh "$STAGE_DIR"
