#!/usr/bin/env bash
#
# Install the Vulkan-enabled Ollama build over an existing Ollama installation
# on an Intel Mac with an AMD GPU.
#
#   curl -fsSL https://raw.githubusercontent.com/Momin010/ollama-vulkan-macos/vulkan-darwin/vulkan-macos/install.sh | bash
#
# Uninstall (restores the original Ollama binaries):
#
#   curl -fsSL .../install.sh | bash -s -- --uninstall
#
set -euo pipefail

REPO="Momin010/ollama-vulkan-macos"
BRANCH="vulkan-darwin"
SCRIPT_URL="https://raw.githubusercontent.com/$REPO/$BRANCH/vulkan-macos/install.sh"

APP="/Applications/Ollama.app"
APP_RESOURCES="$APP/Contents/Resources"
STOCK_BACKUP="$APP_RESOURCES/ollama.stock-backup"
STOCK_LIB_BACKUP="$APP_RESOURCES/lib.stock-backup"
MARKER="$APP_RESOURCES/.vulkan-macos-version"

WORK_DIR=""
LOCK_DIR=""

# Registered at script scope, not inside install_build(), because an EXIT trap
# fires after that function has returned -- a variable local to it would be out
# of scope by then, and referencing it under set -u aborts the script on the way
# out of an otherwise successful install.
#
# Always returns 0: the trap runs on every exit path, including ones where
# WORK_DIR was never assigned, and a trap that ends on a false test can leak a
# non-zero status out of a script that actually succeeded.
cleanup() {
    [ -n "${WORK_DIR:-}" ] && [ -d "$WORK_DIR" ] && rm -rf "$WORK_DIR"
    [ -n "${LOCK_DIR:-}" ] && [ -d "$LOCK_DIR" ] && rmdir "$LOCK_DIR" 2>/dev/null
    return 0
}
trap cleanup EXIT

SUPPORT_DIR="$HOME/Library/Application Support/ollama-vulkan-macos"
CACHE_DIR="$SUPPORT_DIR/payload"
WATCHDOG_LABEL="com.github.momin010.ollama-vulkan-macos.watchdog"
WATCHDOG_PLIST="$HOME/Library/LaunchAgents/$WATCHDOG_LABEL.plist"
WATCHDOG_LOG="$HOME/Library/Logs/ollama-vulkan-macos.log"

bold=$'\033[1m'; red=$'\033[1;31m'; yellow=$'\033[1;33m'; blue=$'\033[1;34m'; green=$'\033[1;32m'; off=$'\033[0m'
log()  { printf '%s==>%s %s\n' "$blue" "$off" "$*"; }
ok()   { printf '%s  ok%s %s\n' "$green" "$off" "$*"; }
warn() { printf '%swarning:%s %s\n' "$yellow" "$off" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$red" "$off" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight
#
# Each check exists because the failure it prevents is confusing rather than
# obvious: an Apple Silicon user would install this and get *slower* inference,
# and a user with no AMD GPU would get no GPU at all with no explanation.
# ---------------------------------------------------------------------------
preflight() {
    [ "$(uname -s)" = "Darwin" ] || die "this installer is for macOS only"

    local arch
    arch="$(uname -m)"
    if [ "$arch" = "arm64" ]; then
        die "this build is for Intel Macs with AMD GPUs.

Apple Silicon Macs already run Ollama on the GPU through Metal, which is
faster than this Vulkan path. You do not need this, and installing it would
make inference slower. Nothing has been changed."
    fi
    [ "$arch" = "x86_64" ] || die "unsupported architecture: $arch"

    local macos_major
    macos_major="$(sw_vers -productVersion | cut -d. -f1)"
    if [ "$macos_major" -lt 13 ]; then
        warn "macOS $(sw_vers -productVersion) is older than any version this has been tested on"
    fi

    if ! system_profiler SPDisplaysDataType 2>/dev/null | grep -qiE 'AMD|Radeon'; then
        warn "no AMD/Radeon GPU detected on this machine."
        warn "this build only accelerates AMD GPUs; on Intel graphics alone it will fall back to CPU."
        printf '\nContinue anyway? [y/N] '
        read -r reply < /dev/tty || reply=""
        case "$reply" in [yY]*) ;; *) die "aborted" ;; esac
    fi

    [ -d "$APP" ] || die "Ollama.app not found at $APP

This installer patches an existing Ollama installation. Install Ollama first
from https://ollama.com/download, then run this again."

    command -v curl >/dev/null || die "curl is required"
    command -v codesign >/dev/null || die "codesign is required (install Xcode Command Line Tools: xcode-select --install)"
}

# ---------------------------------------------------------------------------
# Mutual exclusion
#
# The watchdog and an interactive run must never operate on the app bundle at
# the same time. Without this, installing while the agent happens to fire gives
# the agent a half-written bundle to inspect, and it draws the wrong conclusion
# from it.
# ---------------------------------------------------------------------------
acquire_lock() {
    mkdir -p "$SUPPORT_DIR"
    local candidate="$SUPPORT_DIR/.lock" i
    for i in $(seq 1 60); do
        if mkdir "$candidate" 2>/dev/null; then
            LOCK_DIR="$candidate"
            return 0
        fi
        sleep 1
    done
    return 1
}

# ---------------------------------------------------------------------------
# Signing
#
# Once this app bundle has been ad-hoc signed, every Mach-O inside it must be
# ad-hoc signed too. A binary carrying Apple's signature and the hardened
# runtime flag is killed outright (SIGKILL, exit 137) when launched from inside
# an ad-hoc bundle, because library validation rejects the mismatch.
#
# This matters most on the *uninstall* path, which puts Apple's original binary
# back. `codesign --deep` does not help: it signs nested bundles and frameworks
# but leaves loose Mach-O files in Resources/ alone, which is exactly where the
# ollama binary lives. It has to be signed explicitly.
# ---------------------------------------------------------------------------
sign_installed_binary() {
    # Copying through a download or a temporary directory attaches attributes
    # that invalidate a signature, so clear them before signing rather than
    # after.
    xattr -c "$APP_RESOURCES/ollama" 2>/dev/null || true
    codesign --force --sign - "$APP_RESOURCES/ollama" >/dev/null 2>&1 \
        || warn "could not re-sign the ollama binary; it may be killed on launch"
}

verify_binary_runs() {
    if ! "$APP_RESOURCES/ollama" --version >/dev/null 2>&1; then
        local rc=$?
        if [ "$rc" -eq 137 ]; then
            warn "the installed binary is being killed by macOS (SIGKILL)."
            warn "this is a code-signing mismatch inside the app bundle."
        else
            warn "the installed ollama binary exited $rc when run"
        fi
        return 1
    fi
    return 0
}

# Stop the desktop app *and* its server, and do not return until both are
# actually gone.
#
# The app supervises the server process and restarts it when it dies. Killing
# only the server, or asking the app to quit and assuming it did, leaves a
# supervisor running that respawns a server from whatever is on disk at that
# instant -- in the middle of replacing the binary. The result is a stale
# server running the previous build while the new one sits installed and
# unused, which looks exactly like "the install did nothing".
quit_ollama() {
    log "stopping Ollama"

    # Ask nicely first so the app can shut its server down cleanly.
    osascript -e 'quit app "Ollama"' 2>/dev/null || true

    local i
    for i in $(seq 1 20); do
        pgrep -f "Ollama.app/Contents/MacOS/Ollama" >/dev/null 2>&1 || break
        sleep 0.5
    done

    # Kill the supervisor before the server, otherwise it restarts it.
    if pgrep -f "Ollama.app/Contents/MacOS/Ollama" >/dev/null 2>&1; then
        pkill -f "Ollama.app/Contents/MacOS/Ollama" 2>/dev/null || true
        sleep 2
    fi
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 1

    # Escalate to SIGKILL for anything still holding on.
    for i in $(seq 1 10); do
        pgrep -f "Ollama.app/Contents/MacOS/Ollama|ollama serve" >/dev/null 2>&1 || return 0
        pkill -9 -f "Ollama.app/Contents/MacOS/Ollama" 2>/dev/null || true
        pkill -9 -f "ollama serve" 2>/dev/null || true
        sleep 1
    done

    if pgrep -f "Ollama.app/Contents/MacOS/Ollama|ollama serve" >/dev/null 2>&1; then
        warn "Ollama is still running; quit it manually and run this again"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
uninstall() {
    [ -f "$STOCK_BACKUP" ] || die "no backup found at $STOCK_BACKUP -- nothing to restore.

If Ollama is misbehaving, reinstalling it from https://ollama.com/download
will restore the official build."

    quit_ollama || die "could not stop Ollama; nothing has been changed"

    log "restoring the original Ollama binary"
    cp "$STOCK_BACKUP" "$APP_RESOURCES/ollama"
    rm -f "$STOCK_BACKUP"

    if [ -d "$STOCK_LIB_BACKUP" ]; then
        rm -rf "$APP_RESOURCES/lib"
        mv "$STOCK_LIB_BACKUP" "$APP_RESOURCES/lib"
    else
        # No stock lib existed before we installed one.
        rm -rf "$APP_RESOURCES/lib"
    fi

    rm -f "$MARKER"

    log "re-signing $APP"
    codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || warn "re-signing reported a problem"
    # Must come after the bundle: the restored binary carries Apple's signature
    # and the hardened runtime, which macOS kills inside an ad-hoc bundle.
    sign_installed_binary

    if verify_binary_runs; then
        ok "restored binary runs"
    else
        warn "the restored Ollama binary does not run on this machine."
        warn "reinstalling Ollama from https://ollama.com/download will fix it."
    fi

    remove_watchdog
    rm -rf "$SUPPORT_DIR"

    ok "restored the official Ollama build"
    cat <<EOF

Relaunch Ollama from Applications when you are ready.

Note: this app bundle was ad-hoc re-signed when the Vulkan build was
installed, and that cannot be undone from here. The official Ollama code is
back, but if you want a fully notarized app, reinstall it from
https://ollama.com/download.
EOF
}

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
resolve_release() {
    log "looking up the latest release" >&2
    local api="https://api.github.com/repos/$REPO/releases/latest"
    local json
    json="$(curl -fsSL "$api")" || die "could not reach GitHub to find a release"

    TARBALL_URL="$(printf '%s' "$json" | grep -o '"browser_download_url": *"[^"]*ollama-vulkan-macos-[^"]*\.tar\.gz"' | head -1 | cut -d'"' -f4)"
    RELEASE_TAG="$(printf '%s' "$json" | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4)"

    [ -n "$TARBALL_URL" ] || die "no macOS Vulkan build found in the latest release of $REPO"
}

install_build() {
    resolve_release

    # Deliberately not `local`: the EXIT trap runs after this function has
    # returned, so a local would be out of scope by then -- which under set -u
    # aborts the script on the way out, after a successful install, and leaks
    # the ~200 MB download.
    WORK_DIR="$(mktemp -d)"

    log "downloading $RELEASE_TAG"
    curl -fsSL --progress-bar "$TARBALL_URL" -o "$WORK_DIR/payload.tar.gz" \
        || die "download failed"

    # Verify the checksum when the release publishes one.
    local sums_url="${TARBALL_URL%/*}/checksums.txt"
    if curl -fsSL "$sums_url" -o "$WORK_DIR/checksums.txt" 2>/dev/null; then
        log "verifying checksum"
        local want got
        want="$(grep "$(basename "$TARBALL_URL")" "$WORK_DIR/checksums.txt" | awk '{print $1}')"
        got="$(shasum -a 256 "$WORK_DIR/payload.tar.gz" | awk '{print $1}')"
        if [ -n "$want" ] && [ "$want" != "$got" ]; then
            die "checksum mismatch -- refusing to install.
  expected $want
  got      $got"
        fi
        ok "checksum verified"
    else
        warn "no checksums.txt in the release; skipping verification"
    fi

    log "extracting"
    mkdir -p "$WORK_DIR/payload"
    tar -xzf "$WORK_DIR/payload.tar.gz" -C "$WORK_DIR/payload"

    local src
    src="$WORK_DIR/payload"
    # Tolerate an extra top-level directory in the archive.
    if [ ! -f "$src/ollama" ] && [ "$(find "$src" -maxdepth 1 -type d | wc -l)" -eq 2 ]; then
        src="$(find "$src" -maxdepth 1 -mindepth 1 -type d)"
    fi
    [ -f "$src/ollama" ] || die "archive did not contain an ollama binary"
    [ -d "$src/lib/ollama" ] || die "archive did not contain a lib/ollama payload"

    # Installing underneath a running supervisor is how a stale server ends up
    # serving the old build from a new binary, so this is fatal, not a warning.
    quit_ollama || die "could not stop Ollama; nothing has been changed"

    # Back up the stock binary, but only the first time. Running the installer
    # twice must not overwrite the pristine backup with an already-patched
    # binary -- that would silently destroy the ability to uninstall.
    if [ ! -f "$STOCK_BACKUP" ]; then
        log "backing up the original Ollama binary"
        cp -p "$APP_RESOURCES/ollama" "$STOCK_BACKUP"
        if [ -d "$APP_RESOURCES/lib" ]; then
            cp -R "$APP_RESOURCES/lib" "$STOCK_LIB_BACKUP"
        fi
        ok "original saved to $(basename "$STOCK_BACKUP")"
    else
        log "existing backup found; leaving it untouched"
    fi

    log "installing the Vulkan build"
    rm -rf "$APP_RESOURCES/lib/ollama"
    mkdir -p "$APP_RESOURCES/lib"
    cp -R "$src/lib/ollama" "$APP_RESOURCES/lib/ollama"
    cp "$src/ollama" "$APP_RESOURCES/ollama"
    chmod +x "$APP_RESOURCES/ollama"

    # Files downloaded by curl carry a quarantine attribute that makes
    # Gatekeeper refuse to load them.
    log "clearing quarantine attributes"
    xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

    log "re-signing $APP"
    codesign --force --deep --sign - "$APP" >/dev/null 2>&1 \
        || warn "re-signing reported a problem; the app may refuse to launch"
    sign_installed_binary

    verify_binary_runs || die "the installed binary will not run on this machine.
Restoring the official build:
  curl -fsSL $SCRIPT_URL | bash -s -- --uninstall"

    printf '%s\n' "$RELEASE_TAG" > "$MARKER"

    # Keep a local copy so the patch can be re-applied without a download,
    # which is what makes the watchdog able to work offline and instantly.
    log "caching payload for repair"
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"
    cp -R "$src/lib" "$CACHE_DIR/lib"
    cp "$src/ollama" "$CACHE_DIR/ollama"
    printf '%s\n' "$RELEASE_TAG" > "$SUPPORT_DIR/version"

    ok "installed $RELEASE_TAG"
}

# ---------------------------------------------------------------------------
# Repair: re-apply the cached payload if something replaced it.
#
# Ollama's auto-update overwrites the patched binary with the official one.
# There is no error when this happens -- inference silently moves back to the
# CPU -- so the failure is easy to miss and hard to attribute.
# ---------------------------------------------------------------------------
needs_repair() {
    [ -f "$CACHE_DIR/ollama" ] || return 1
    [ -f "$APP_RESOURCES/ollama" ] || return 0
    [ -f "$MARKER" ] || return 0

    # cmp's exit status has three meanings and they must not be conflated:
    #   0  identical            -> nothing to do
    #   1  differ               -> repair
    #  >1  cmp itself failed    -> we do not know
    #
    # The third case is real: during an install the bundle is being rewritten
    # underneath us and cmp has been observed dying on SIGKILL. Treating "could
    # not tell" as "differs" makes the watchdog perform a destructive repair
    # against a half-written bundle, so an unreliable answer must mean "do
    # nothing".
    local rc=0
    cmp -s "$CACHE_DIR/ollama" "$APP_RESOURCES/ollama" || rc=$?
    case "$rc" in
        0) return 1 ;;
        1) return 0 ;;
        *) log "could not compare binaries (cmp exited $rc); leaving things alone"
           return 1 ;;
    esac
}

repair() {
    [ -d "$APP" ] || { log "Ollama.app not present; nothing to repair"; return 0; }

    if ! needs_repair; then
        log "Vulkan build is intact; nothing to do"
        return 0
    fi

    log "Ollama was replaced (likely by an auto-update); re-applying the Vulkan build"
    quit_ollama

    # The stock backup is deliberately NOT refreshed here.
    #
    # It is tempting to save the binary that just replaced ours, on the grounds
    # that a newer official build is what a later uninstall should restore. But
    # this code path runs unattended, and if it ever misfires it overwrites the
    # only pristine copy of the official binary with a patched one -- silently
    # destroying the user's ability to uninstall. That risk is not worth a
    # version bump, so the backup is written exactly once, at install time.
    # Stage first, swap last.
    #
    # An earlier version removed lib/ollama and then copied the replacement in.
    # Running from a LaunchAgent that copy fails with "Operation not permitted"
    # -- a background agent does not inherit the rights needed to write inside
    # /Applications -- and the delete had already happened. The result was an
    # empty lib/ollama, no Vulkan backend to load, and silent CPU inference: the
    # exact failure this watchdog exists to prevent, caused by the watchdog.
    #
    # Nothing is now destroyed until a full copy has been written and verified
    # alongside it.
    local staged="$APP_RESOURCES/lib/.ollama-staging.$$"
    rm -rf "$staged"
    if ! mkdir -p "$APP_RESOURCES/lib" 2>/dev/null; then
        warn "cannot write to $APP_RESOURCES/lib; leaving the installation untouched"
        warn "re-run the installer from a terminal to repair it"
        return 1
    fi
    if ! cp -R "$CACHE_DIR/lib/ollama" "$staged" 2>/dev/null; then
        rm -rf "$staged"
        warn "could not stage the runtime payload (permission denied?);"
        warn "leaving the installation untouched -- re-run the installer from a terminal"
        return 1
    fi

    # A staged payload missing its backend would be worse than doing nothing.
    if [ ! -s "$staged/vulkan/libggml-vulkan.so" ]; then
        rm -rf "$staged"
        warn "staged payload is incomplete; leaving the installation untouched"
        return 1
    fi

    local staged_bin="$APP_RESOURCES/.ollama-staging.$$"
    if ! cp "$CACHE_DIR/ollama" "$staged_bin" 2>/dev/null; then
        rm -rf "$staged" "$staged_bin"
        warn "could not stage the ollama binary; leaving the installation untouched"
        return 1
    fi

    # Both pieces are present and writable. Now swap.
    rm -rf "$APP_RESOURCES/lib/ollama"
    mv "$staged" "$APP_RESOURCES/lib/ollama"
    mv "$staged_bin" "$APP_RESOURCES/ollama"
    chmod +x "$APP_RESOURCES/ollama"
    xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
    codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || warn "re-signing reported a problem"
    sign_installed_binary
    cp "$SUPPORT_DIR/version" "$MARKER" 2>/dev/null || true

    verify_binary_runs || warn "the re-applied binary does not run; run the installer again"

    if [ ! -s "$APP_RESOURCES/lib/ollama/vulkan/libggml-vulkan.so" ]; then
        warn "the Vulkan backend is missing after repair -- inference will fall back to CPU"
        warn "re-run the installer from a terminal"
    fi

    local restored
    restored="$(cat "$SUPPORT_DIR/version" 2>/dev/null || echo 'the Vulkan build')"
    ok "re-applied $restored"
    # The cached build corresponds to whichever upstream release this fork was
    # tracking when it was installed. If Ollama updated to something newer,
    # restoring the cache keeps GPU support but pins the older Ollama version.
    log "note: this restores Ollama $restored; re-run the installer to pick up a newer build"
    open "$APP" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------
install_watchdog() {
    mkdir -p "$SUPPORT_DIR" "$(dirname "$WATCHDOG_PLIST")" "$(dirname "$WATCHDOG_LOG")"

    # The agent needs a local copy of this script. On the primary install path
    # (curl | bash) the script is read from stdin and there is no file on disk
    # to copy, so fall back to downloading it.
    local self="${BASH_SOURCE[0]:-}"
    if [ -n "$self" ] && [ "$self" != "bash" ] && [ -f "$self" ]; then
        cp "$self" "$SUPPORT_DIR/install.sh"
    else
        curl -fsSL "$SCRIPT_URL" -o "$SUPPORT_DIR/install.sh" \
            || { warn "could not download the repair script; skipping watchdog"; return 1; }
    fi
    chmod +x "$SUPPORT_DIR/install.sh"

    cat > "$WATCHDOG_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$WATCHDOG_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$SUPPORT_DIR/install.sh</string>
        <string>--repair</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StartInterval</key>
    <integer>21600</integer>
    <key>StandardOutPath</key>
    <string>$WATCHDOG_LOG</string>
    <key>StandardErrorPath</key>
    <string>$WATCHDOG_LOG</string>
</dict>
</plist>
EOF

    launchctl unload "$WATCHDOG_PLIST" 2>/dev/null || true
    launchctl load "$WATCHDOG_PLIST" 2>/dev/null \
        || { warn "could not load the watchdog agent"; return 1; }

    ok "watchdog installed (checks at login and every 6 hours)"
}

remove_watchdog() {
    [ -f "$WATCHDOG_PLIST" ] || return 0
    launchctl unload "$WATCHDOG_PLIST" 2>/dev/null || true
    rm -f "$WATCHDOG_PLIST"
    ok "watchdog removed"
}

verify_gpu() {
    local logf="$HOME/.ollama/logs/server.log"

    # Only read what the server we are about to start writes. The log persists
    # across runs, so matching anywhere in the file can report a Vulkan device
    # discovered by a previous install -- or, as happened during testing, miss
    # that the line at the end came from a stale server.
    local from=0
    [ -f "$logf" ] && from="$(wc -l < "$logf" | tr -d ' ')"

    log "starting Ollama and checking GPU discovery"
    open "$APP" 2>/dev/null || { warn "could not launch Ollama automatically"; return 0; }

    local i line=""
    for i in $(seq 1 45); do
        sleep 1
        curl -fsS -m 2 localhost:11434/api/version >/dev/null 2>&1 || continue
        [ -f "$logf" ] || continue
        line="$(tail -n "+$((from + 1))" "$logf" 2>/dev/null | grep 'inference compute' | tail -1)"
        [ -n "$line" ] && break
    done

    if [ -z "$line" ]; then
        warn "Ollama did not report a compute device within 45s."
        warn "check: grep 'inference compute' $logf"
        return 0
    fi

    if printf '%s' "$line" | grep -q 'library=Vulkan'; then
        ok "GPU in use: $(printf '%s' "$line" | sed -n 's/.*description="\([^"]*\)".*/\1/p')"
        return 0
    fi

    warn "Ollama started but is NOT using the GPU."
    warn "it reported: $(printf '%s' "$line" | grep -oE 'library=[a-zA-Z]+')"
    warn "quit Ollama completely from the menu bar and reopen it, then check:"
    warn "  grep 'inference compute' $logf"
    return 0
}

usage() {
    cat <<EOF
Ollama with Vulkan GPU acceleration for Intel Macs with AMD GPUs.

Usage: install.sh [option]

  (no option)   install or update the Vulkan build
  --uninstall   restore the official Ollama build and remove everything
  --repair      re-apply the cached build if an Ollama update replaced it
  --no-watchdog install without the background auto-repair agent
  --help        show this message

EOF
}

main() {
    local want_watchdog=1

    case "${1:-}" in
        --uninstall)
            preflight
            acquire_lock || die "another install or repair is in progress"
            uninstall
            return
            ;;
        --repair)
            # Runs unattended from the watchdog: no prompts, no preflight
            # that could block on a terminal that is not there.
            printf '\n[%s] repair check\n' "$(date '+%Y-%m-%d %H:%M:%S')"
            # If an interactive install is running, that operation owns the
            # bundle. Exiting quietly is correct: the installer leaves things
            # in the state this would have tried to produce anyway.
            if ! acquire_lock; then
                log "another install or repair is in progress; skipping"
                return 0
            fi
            repair
            return
            ;;
        --no-watchdog)
            want_watchdog=0
            ;;
        --help|-h)
            usage
            return
            ;;
        "")
            ;;
        *)
            printf '%serror:%s unknown option: %s\n\n' "$red" "$off" "$1" >&2
            usage >&2
            exit 1
            ;;
    esac

    printf '\n%sOllama + Vulkan for Intel Macs with AMD GPUs%s\n\n' "$bold" "$off"
    preflight
    acquire_lock || die "another install or repair is in progress"
    install_build
    if [ "$want_watchdog" -eq 1 ]; then
        install_watchdog || warn "continuing without the watchdog"
    fi
    verify_gpu

    cat <<EOF

${bold}Done.${off}

  Ollama now runs models on your AMD GPU instead of the CPU.
  Everything else about Ollama works exactly as before.

  ${yellow}One thing to know:${off} if Ollama updates itself, the update replaces
  this build with the official one and you are quietly back on CPU.
$(if [ -f "$WATCHDOG_PLIST" ]; then
    printf '  The watchdog checks for this at login and every 6 hours, and\n'
    printf '  puts the Vulkan build back automatically.\n'
  else
    printf '  Re-run this installer if that happens.\n'
  fi)

  Check which device is in use:  grep 'inference compute' ~/.ollama/logs/server.log
  Measure speed:                 ollama run llama3.2 --verbose
  Revert to the official build:  curl -fsSL $SCRIPT_URL | bash -s -- --uninstall

EOF
}

main "$@"
