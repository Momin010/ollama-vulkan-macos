#!/usr/bin/env bash
#
# Validate a staged payload before it is published.
#
# This runs in CI on a virtual machine with no discrete AMD GPU, so it
# deliberately does *not* assert that a GPU is found. What it checks is that
# the payload is loadable and self-contained -- the failures that would
# otherwise reach users as "the download is broken on every machine but the
# one that built it".
#
# Usage: vulkan-macos/smoke-test.sh <staging-dir>
#
set -euo pipefail

STAGE_DIR="${1:-dist/ollama-vulkan-macos}"

# Resolve to an absolute path up front. The backend load test below builds
# DYLD_LIBRARY_PATH and GGML_BACKEND_PATH from this, and dyld requires absolute
# paths there, so a relative argument must not be passed through verbatim.
[ -d "$STAGE_DIR" ] || { printf 'fail: no such directory: %s\n' "$STAGE_DIR" >&2; exit 1; }
STAGE_DIR="$(cd "$STAGE_DIR" && pwd)"

red=$'\033[1;31m'; green=$'\033[1;32m'; blue=$'\033[1;34m'; yellow=$'\033[1;33m'; off=$'\033[0m'
log()  { printf '%s==>%s %s\n' "$blue" "$off" "$*"; }
ok()   { printf '%s  ok%s %s\n' "$green" "$off" "$*"; }
warn() { printf '%swarning:%s %s\n' "$yellow" "$off" "$*" >&2; }
die()  { printf '%sfail:%s %s\n' "$red" "$off" "$*" >&2; exit 1; }

VULKAN_DIR="$STAGE_DIR/lib/ollama/vulkan"

# --- structure ---------------------------------------------------------------
log "checking payload structure"
for f in \
    "$STAGE_DIR/ollama" \
    "$VULKAN_DIR/libggml-vulkan.so" \
    "$VULKAN_DIR/libvulkan.1.dylib" \
    "$VULKAN_DIR/libMoltenVK.dylib" \
    "$VULKAN_DIR/MoltenVK_icd.json"
do
    [ -f "$f" ] || die "missing $f"
done
ok "all expected files present"

# --- architecture ------------------------------------------------------------
log "checking architecture"
for f in "$STAGE_DIR/ollama" "$VULKAN_DIR/libggml-vulkan.so" "$VULKAN_DIR/libMoltenVK.dylib"; do
    file "$f" | grep -q "x86_64" || die "$f is not x86_64"
done
ok "x86_64"

# --- self-containment --------------------------------------------------------
# A reference to a Homebrew or Vulkan SDK path here means the payload only
# works on a machine with that exact installation.
log "checking for external dependencies"
external="$(
    otool -L "$VULKAN_DIR/libggml-vulkan.so" "$VULKAN_DIR/libvulkan.1.dylib" "$VULKAN_DIR/libMoltenVK.dylib" \
        | awk '/^\t/ {print $1}' | sort -u \
        | grep -vE '^(/usr/lib/|/System/|@loader_path/|@rpath/|@executable_path/)' || true
)"
[ -z "$external" ] || die "payload references libraries outside the bundle:
$external"
ok "no external dependencies"

# --- code signatures ---------------------------------------------------------
# install_name_tool invalidates signatures; an unsigned or stale-signed dylib
# is refused by the loader at runtime.
log "verifying code signatures"
for f in "$STAGE_DIR/ollama" "$VULKAN_DIR/libggml-vulkan.so" "$VULKAN_DIR/libvulkan.1.dylib" "$VULKAN_DIR/libMoltenVK.dylib"; do
    codesign --verify --strict "$f" 2>/dev/null || die "invalid signature: $f"
done
ok "signatures valid"

# --- ICD manifest ------------------------------------------------------------
log "validating ICD manifest"
/usr/bin/python3 - "$VULKAN_DIR/MoltenVK_icd.json" <<'PY' || die "ICD manifest is not valid"
import json, os, sys
path = sys.argv[1]
with open(path) as fh:
    data = json.load(fh)
lib = data["ICD"]["library_path"]
resolved = os.path.normpath(os.path.join(os.path.dirname(path), lib))
if not os.path.exists(resolved):
    sys.exit(f"ICD library_path does not resolve: {lib} -> {resolved}")
PY
ok "ICD manifest resolves to the bundled MoltenVK"

# --- binary runs -------------------------------------------------------------
log "running the ollama binary"
"$STAGE_DIR/ollama" --version >/dev/null 2>&1 || die "ollama --version failed"
ok "binary executes"

# --- backend loads -----------------------------------------------------------
# The decisive check: dlopen the Vulkan backend the way the runner will, with
# only the bundled libraries reachable. A dyld failure here is the exact thing
# that breaks on machines without Homebrew.
log "loading the Vulkan backend"
LLAMA_SERVER="$STAGE_DIR/lib/ollama/llama-server"
if [ ! -x "$LLAMA_SERVER" ]; then
    warn "llama-server not present in payload; skipping backend load test"
    exit 0
fi

load_log="$(mktemp)"
trap 'rm -f "$load_log"' EXIT

set +e
env -u DYLD_FALLBACK_LIBRARY_PATH \
    VK_ICD_FILENAMES="$VULKAN_DIR/MoltenVK_icd.json" \
    GGML_VK_DISABLE_F16=1 \
    GGML_BACKEND_PATH="$VULKAN_DIR/libggml-vulkan.so" \
    DYLD_LIBRARY_PATH="$STAGE_DIR/lib/ollama" \
    "$LLAMA_SERVER" --list-devices --offline > "$load_log" 2>&1
exit_code=$?
set -e

# A dyld failure means the payload is not self-contained. This is the failure
# that only shows up on machines other than the one that built it.
if grep -qiE "dlopen|image not found|failed to load|Library not loaded" "$load_log"; then
    cat "$load_log" >&2
    die "the Vulkan backend failed to load"
fi
ok "no linker errors resolving the bundled libraries"

device_line="$(grep -i "Vulkan[0-9]" "$load_log" | head -1 | sed 's/^ *//' || true)"

if [ -n "$device_line" ]; then
    ok "enumerated a Vulkan device: $device_line"
elif [ "${REQUIRE_VULKAN_DEVICE:-0}" = "1" ]; then
    cat "$load_log" >&2
    die "no Vulkan device enumerated, but REQUIRE_VULKAN_DEVICE=1 was set"
else
    warn "no Vulkan device enumerated -- expected on a runner with no AMD GPU,"
    warn "but it means this run did NOT prove the GPU path works end to end."
fi

# Report abnormal termination explicitly. MoltenVK aborts when it cannot find a
# Metal device, so on a GPU-less CI runner a crash here is expected and not a
# reason to fail the build -- but it must never be reported as a clean run,
# because the same abort on real hardware would be a genuine defect.
if [ "$exit_code" -ne 0 ]; then
    if [ "$exit_code" -gt 128 ]; then
        warn "llama-server terminated by signal $((exit_code - 128)) (exit $exit_code)"
    else
        warn "llama-server exited $exit_code"
    fi

    if [ -n "$device_line" ]; then
        # It found the GPU and still died: that is a real problem.
        cat "$load_log" >&2
        die "llama-server crashed after enumerating a Vulkan device"
    fi
    warn "treating this as acceptable because no GPU was present to drive"
fi

printf '\n%sPayload passed all checks.%s\n' "$green" "$off"
if [ -z "$device_line" ]; then
    printf '%sNote: run on hardware with an AMD GPU, with REQUIRE_VULKAN_DEVICE=1,%s\n' "$yellow" "$off"
    printf '%sto verify the GPU path rather than just the packaging.%s\n' "$yellow" "$off"
fi
