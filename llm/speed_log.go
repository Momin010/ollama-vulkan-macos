package llm

import (
	"fmt"
	"log/slog"
	"os"
	"time"
)

// logGenerationSpeed records tokens per second for a completed request.
//
// Ollama already computes these numbers for its API response but never logs
// them, so the only way to see how fast generation actually ran is to call the
// HTTP API by hand. On a machine where throughput varies a great deal -- this
// GPU loses roughly 40% to thermal throttling within a minute of sustained
// load -- being able to watch the rate is the difference between noticing a
// problem and not.
//
// The line is emitted in a fixed, easily parsed shape so external tools can
// tail it:
//
//	ollama-speed model=llama3.2 prompt_tok=12 prompt_tps=84.9 gen_tok=200 gen_tps=45.7
//
// Set OLLAMA_SPEED_LOG=0 to turn it off.
func logGenerationSpeed(model string, promptCount int, promptDur time.Duration, evalCount int, evalDur time.Duration) {
	if os.Getenv("OLLAMA_SPEED_LOG") == "0" {
		return
	}
	if evalCount <= 0 || evalDur <= 0 {
		return
	}

	genTPS := float64(evalCount) / evalDur.Seconds()

	promptTPS := 0.0
	if promptCount > 0 && promptDur > 0 {
		promptTPS = float64(promptCount) / promptDur.Seconds()
	}

	if model == "" {
		model = "unknown"
	}

	slog.Info("ollama-speed",
		"model", model,
		"prompt_tok", promptCount,
		"prompt_tps", fmt.Sprintf("%.1f", promptTPS),
		"gen_tok", evalCount,
		"gen_tps", fmt.Sprintf("%.1f", genTPS),
	)
}
