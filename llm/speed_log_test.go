package llm

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"time"
)

func captureSpeedLog(t *testing.T, fn func()) string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo})))
	t.Cleanup(func() { slog.SetDefault(prev) })
	fn()
	return buf.String()
}

func TestLogGenerationSpeed(t *testing.T) {
	t.Run("reports both rates", func(t *testing.T) {
		t.Setenv("OLLAMA_SPEED_LOG", "")
		out := captureSpeedLog(t, func() {
			// 200 tokens in 4s = 50.0 tok/s; 12 prompt tokens in 0.1s = 120.0
			logGenerationSpeed("llama3.2", 12, 100*time.Millisecond, 200, 4*time.Second)
		})
		for _, want := range []string{"ollama-speed", "gen_tps=50.0", "prompt_tps=120.0", "gen_tok=200"} {
			if !strings.Contains(out, want) {
				t.Errorf("missing %q in: %s", want, out)
			}
		}
	})

	t.Run("disabled by env", func(t *testing.T) {
		t.Setenv("OLLAMA_SPEED_LOG", "0")
		out := captureSpeedLog(t, func() {
			logGenerationSpeed("llama3.2", 12, 100*time.Millisecond, 200, 4*time.Second)
		})
		if strings.Contains(out, "ollama-speed") {
			t.Errorf("should be silent when disabled, got: %s", out)
		}
	})

	// A cancelled or empty generation must not produce a divide-by-zero or a
	// meaningless line.
	t.Run("silent when nothing was generated", func(t *testing.T) {
		t.Setenv("OLLAMA_SPEED_LOG", "")
		for _, tc := range []struct {
			name  string
			count int
			dur   time.Duration
		}{
			{"zero tokens", 0, time.Second},
			{"zero duration", 100, 0},
			{"negative duration", 100, -time.Second},
		} {
			out := captureSpeedLog(t, func() {
				logGenerationSpeed("m", 0, 0, tc.count, tc.dur)
			})
			if strings.Contains(out, "ollama-speed") {
				t.Errorf("%s: expected no line, got: %s", tc.name, out)
			}
		}
	})

	t.Run("handles a missing prompt phase", func(t *testing.T) {
		t.Setenv("OLLAMA_SPEED_LOG", "")
		out := captureSpeedLog(t, func() {
			logGenerationSpeed("", 0, 0, 50, time.Second)
		})
		if !strings.Contains(out, "prompt_tps=0.0") {
			t.Errorf("expected zero prompt rate, got: %s", out)
		}
		if !strings.Contains(out, "model=unknown") {
			t.Errorf("expected placeholder model name, got: %s", out)
		}
	})
}
