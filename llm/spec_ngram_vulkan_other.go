//go:build !darwin

package llm

import "github.com/ollama/ollama/ml"

// appendVulkanNgramSpecArgs is a no-op off darwin. The n-gram default it adds
// was measured against Vulkan-on-MoltenVK, and nothing is known about whether
// it helps or hurts on a native Vulkan driver, so it is not applied there.
func appendVulkanNgramSpecArgs(params []string, _ []ml.DeviceInfo) []string {
	return params
}
