//go:build darwin

package llm

import (
	"os"
	"slices"

	"github.com/ollama/ollama/ml"
)

// appendVulkanNgramSpecArgs enables n-gram cache speculative decoding for
// Vulkan devices on macOS.
//
// Token generation here is bound by memory bandwidth: every token reads the
// whole model once. n-gram cache drafting proposes continuations by matching
// against text already in the context rather than by running a second model,
// so the drafts cost no additional weight traffic. Accepted tokens are
// effectively free, and rejected ones cost only the wider verification step.
//
// Measured on an AMD Radeon Pro 5500M with llama3.2 3B Q4_K_M, over two
// independent drift-controlled runs:
//
//	prose, no repetition   45.12 -> 45.18 tok/s   (unchanged)
//	text echoing its input 40.57 -> 48.38 tok/s   (+19%)
//
// The gain appears wherever output repeats the input, which is the common case
// for editing code, refactoring, and summarising a file. It is not a win for
// free-form writing, but it is not a loss either.
//
// Model-based drafting was measured too and is deliberately not used: the
// smallest same-tokenizer draft available for llama3.2 is 1B, which is 1.32 GB
// against a 2.02 GB target. Reading the draft costs nearly as much as the
// verification it saves, and it halved throughput in practice.
//
// Skipped when another speculative mode has already been selected, and
// disabled entirely with OLLAMA_VULKAN_NGRAM=0.
func appendVulkanNgramSpecArgs(params []string, gpus []ml.DeviceInfo) []string {
	if os.Getenv("OLLAMA_VULKAN_NGRAM") == "0" {
		return params
	}

	// Another --spec-type wins; do not fight over the flag.
	if slices.Contains(params, "--spec-type") {
		return params
	}

	var usesVulkan bool
	for _, gpu := range gpus {
		if gpu.Library == "Vulkan" {
			usesVulkan = true
			break
		}
	}
	if !usesVulkan {
		return params
	}

	return append(params, "--spec-type", "ngram-cache")
}
