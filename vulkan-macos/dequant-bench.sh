#!/usr/bin/env bash
# Isolate dequantisation cost.
#
# All models below are the same Llama 3.2 1B weights with the same tensor
# shapes. The only variable is the weight format, i.e. how much ALU work the
# GPU spends unpacking each value. If unpacking cost is what separates 65%
# from 83% of achievable bandwidth, cheaper formats should reach a higher
# fraction of the 141.4 GB/s streaming ceiling at similar size.
set -uo pipefail
SP="$(cd "$(dirname "$0")" && pwd)"; OUT="$SP/dequant-results.txt"; : > "$OUT"
DIST="$HOME/ollama-vulkan-macos/dist/ollama-vulkan-macos"; VK="$DIST/lib/ollama/vulkan"
PORT=11999; ACH=141.4; THEO=192.0
say(){ printf '%s\n' "$*" | tee -a "$OUT"; }

kill_servers(){ pkill -9 -f "llama-server" 2>/dev/null; local i
  for i in $(seq 1 25); do [ -z "$(pgrep -f llama-server)" ] && [ -z "$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null)" ] && return 0; sleep 1; done; return 1; }

start(){ kill_servers || { echo ""; return; }
  env -u DYLD_FALLBACK_LIBRARY_PATH VK_ICD_FILENAMES="$VK/MoltenVK_icd.json" \
      GGML_BACKEND_PATH="$VK/libggml-vulkan.so" DYLD_LIBRARY_PATH="$DIST/lib/ollama" \
      GGML_VK_DISABLE_F16=1 \
      "$DIST/lib/ollama/llama-server" --model "$1" --port $PORT --host 127.0.0.1 \
        --n-gpu-layers 99 --ctx-size 4096 --no-webui --offline > /tmp/ls-dq.log 2>&1 &
  local pid=$! i
  for i in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || { echo ""; return; }
    if curl -fsS -m 2 "localhost:$PORT/health" >/dev/null 2>&1; then
      local owner; owner="$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null | head -1)"
      [ "$owner" = "$pid" ] && { echo "$pid"; return; }; echo ""; return
    fi; sleep 1
  done; echo ""; }

measure(){ curl -s -m 600 "localhost:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Write a detailed paragraph about the ocean and its currents.","n_predict":200,"temperature":0}' \
    | /usr/bin/python3 -c "import json,sys
try: print('%.2f'%json.load(sys.stdin)['timings']['predicted_per_second'])
except Exception: print('0')" 2>/dev/null; }

bench(){ # label file
  sleep 75
  local pid; pid="$(start "$2")"
  if [ -z "$pid" ]; then say "$(printf '  %-12s SERVER FAILED' "$1")"; return; fi
  measure >/dev/null
  local t1 t2 t; t1="$(measure)"; t2="$(measure)"
  t="$(/usr/bin/python3 -c "print(max(float('$t1'),float('$t2')))")"
  local gb; gb="$(/usr/bin/python3 -c "import os;print(os.path.getsize('$2')/1e9)")"
  /usr/bin/python3 -c "
t=float('$t'); gb=float('$gb'); bw=t*gb
print('  %-12s %6.3f GB  %7.2f tok/s  %7.1f GB/s  %3.0f%% of 192  %3.0f%% of 141.4'
      %('$1',gb,t,bw,100*bw/$THEO,100*bw/$ACH))" | tee -a "$OUT"
  kill_servers; }

say "=== dequantisation cost, isolated (Llama 3.2 1B, identical shapes) ==="
say "  reference: same model as f16 2.48GB -> 117.9 GB/s = 83% of achievable"
say ""
kill_servers; say "cooling 150s"; sleep 150

bench "Q4_0"   "$SP/quants/1b-Q4_0.gguf"
bench "Q4_K_M" "$SP/quants/1b-Q4_K_M.gguf"
bench "Q4_1"   "$SP/quants/1b-Q4_1.gguf"
bench "IQ4_NL" "$SP/quants/1b-IQ4_NL.gguf"

say ""
say "Q4_0 vs Q4_K_M is the key pair: near-identical size, far cheaper unpacking."
say "=== done ==="
