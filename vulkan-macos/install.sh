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

quit_ollama() {
    log "stopping Ollama"
    osascript -e 'quit app "Ollama"' 2>/dev/null || true
    pkill -f "ollama serve" 2>/dev/null || true
    # Give the app time to release the binary before it is replaced.
    local i
    for i in $(seq 1 20); do
        pgrep -f "Ollama.app" >/dev/null 2>&1 || break
        sleep 0.5
    done
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
uninstall() {
    [ -f "$STOCK_BACKUP" ] || die "no backup found at $STOCK_BACKUP -- nothing to restore.

If Ollama is misbehaving, reinstalling it from https://ollama.com/download
will restore the official build."

    quit_ollama

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

    remove_watchdog
    rm -rf "$SUPPORT_DIR"

    ok "restored the official Ollama build"
    printf '\nRelaunch Ollama from Applications when you are ready.\n'
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

    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT

    log "downloading $RELEASE_TAG"
    curl -fsSL --progress-bar "$TARBALL_URL" -o "$tmp/payload.tar.gz" \
        || die "download failed"

    # Verify the checksum when the release publishes one.
    local sums_url="${TARBALL_URL%/*}/checksums.txt"
    if curl -fsSL "$sums_url" -o "$tmp/checksums.txt" 2>/dev/null; then
        log "verifying checksum"
        local want got
        want="$(grep "$(basename "$TARBALL_URL")" "$tmp/checksums.txt" | awk '{print $1}')"
        got="$(shasum -a 256 "$tmp/payload.tar.gz" | awk '{print $1}')"
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
    mkdir -p "$tmp/payload"
    tar -xzf "$tmp/payload.tar.gz" -C "$tmp/payload"

    local src
    src="$tmp/payload"
    # Tolerate an extra top-level directory in the archive.
    if [ ! -f "$src/ollama" ] && [ "$(find "$src" -maxdepth 1 -type d | wc -l)" -eq 2 ]; then
        src="$(find "$src" -maxdepth 1 -mindepth 1 -type d)"
    fi
    [ -f "$src/ollama" ] || die "archive did not contain an ollama binary"
    [ -d "$src/lib/ollama" ] || die "archive did not contain a lib/ollama payload"

    quit_ollama

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
    # Marker gone, or the installed binary no longer matches the cached one.
    [ -f "$MARKER" ] || return 0
    ! cmp -s "$CACHE_DIR/ollama" "$APP_RESOURCES/ollama"
}

repair() {
    [ -d "$APP" ] || { log "Ollama.app not present; nothing to repair"; return 0; }

    if ! needs_repair; then
        log "Vulkan build is intact; nothing to do"
        return 0
    fi

    log "Ollama was replaced (likely by an auto-update); re-applying the Vulkan build"
    quit_ollama

    # Refresh the stock backup: the binary that just overwrote ours is a
    # newer official build, and is the correct thing to restore on uninstall.
    if [ -f "$APP_RESOURCES/ollama" ] && ! cmp -s "$CACHE_DIR/ollama" "$APP_RESOURCES/ollama"; then
        cp -p "$APP_RESOURCES/ollama" "$STOCK_BACKUP"
    fi

    rm -rf "$APP_RESOURCES/lib/ollama"
    mkdir -p "$APP_RESOURCES/lib"
    cp -R "$CACHE_DIR/lib/ollama" "$APP_RESOURCES/lib/ollama"
    cp "$CACHE_DIR/ollama" "$APP_RESOURCES/ollama"
    chmod +x "$APP_RESOURCES/ollama"
    xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
    codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || warn "re-signing reported a problem"
    cp "$SUPPORT_DIR/version" "$MARKER" 2>/dev/null || true

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
    log "starting Ollama and checking GPU discovery"
    open "$APP" 2>/dev/null || { warn "could not launch Ollama automatically"; return 0; }

    local i
    for i in $(seq 1 30); do
        curl -fsS -m 2 localhost:11434/api/version >/dev/null 2>&1 && break
        sleep 1
    done

    local logf="$HOME/.ollama/logs/server.log"
    if [ ! -f "$logf" ]; then
        warn "no server log yet; open Ollama and try a prompt to confirm"
        return 0
    fi

    local line
    line="$(grep 'inference compute' "$logf" 2>/dev/null | tail -1)"
    if printf '%s' "$line" | grep -q 'library=Vulkan'; then
        ok "GPU in use: $(printf '%s' "$line" | sed -n 's/.*description="\([^"]*\)".*/\1/p')"
    else
        warn "Ollama started but the log does not show a Vulkan device."
        warn "check: grep 'inference compute' $logf"
    fi
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
            uninstall
            return
            ;;
        --repair)
            # Runs unattended from the watchdog: no prompts, no preflight
            # that could block on a terminal that is not there.
            printf '\n[%s] repair check\n' "$(date '+%Y-%m-%d %H:%M:%S')"
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
