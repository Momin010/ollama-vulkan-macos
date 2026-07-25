//go:build darwin

package llm

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

// TestLlamaServerFlashAttentionVulkanDarwin pins the invariant that flash
// attention is never enabled for Vulkan devices on macOS, including when the
// user has explicitly asked for it. Under MoltenVK the flash-attention tensor
// is assigned to the CPU backend, which forces a GPU->CPU round trip per token
// and collapses generation from ~35 tok/s to under 1 tok/s.
func TestLlamaServerFlashAttentionVulkanDarwin(t *testing.T) {
	vulkan := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}}
	metal := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}}

	cases := []struct {
		name    string
		envFA   string
		gpus    []ml.DeviceInfo
		want    ml.FlashAttentionType
		exactly bool
	}{
		{
			name:    "vulkan disables fa",
			gpus:    []ml.DeviceInfo{vulkan},
			want:    ml.FlashAttentionDisabled,
			exactly: true,
		},
		{
			name:    "vulkan disables fa even when explicitly enabled",
			envFA:   "1",
			gpus:    []ml.DeviceInfo{vulkan},
			want:    ml.FlashAttentionDisabled,
			exactly: true,
		},
		{
			name:    "mixed device set with any vulkan disables fa",
			envFA:   "1",
			gpus:    []ml.DeviceInfo{metal, vulkan},
			want:    ml.FlashAttentionDisabled,
			exactly: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("OLLAMA_FLASH_ATTENTION", tc.envFA)
			if got := LlamaServerFlashAttention(tc.gpus); got != tc.want {
				t.Errorf("LlamaServerFlashAttention() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestLlamaServerFlashAttentionNonVulkanUnaffected ensures the darwin override
// is scoped to Vulkan and does not change behaviour for Metal, which is what
// Apple Silicon Macs use and where flash attention works correctly.
func TestLlamaServerFlashAttentionNonVulkanUnaffected(t *testing.T) {
	t.Setenv("OLLAMA_FLASH_ATTENTION", "1")

	metal := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}}}
	if got := LlamaServerFlashAttention(metal); got == ml.FlashAttentionDisabled {
		t.Error("Metal devices should not be forced to FlashAttentionDisabled by the Vulkan override")
	}
}
