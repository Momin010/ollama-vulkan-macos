//go:build darwin

package llm

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestAppendVulkanNgramSpecArgs(t *testing.T) {
	vulkan := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}}}
	metal := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}}}

	t.Run("enabled for vulkan", func(t *testing.T) {
		t.Setenv("OLLAMA_VULKAN_NGRAM", "")
		got := appendVulkanNgramSpecArgs(nil, vulkan)
		if !slices.Contains(got, "ngram-cache") {
			t.Errorf("expected ngram-cache to be enabled, got %v", got)
		}
	})

	t.Run("not applied to non-vulkan devices", func(t *testing.T) {
		t.Setenv("OLLAMA_VULKAN_NGRAM", "")
		if got := appendVulkanNgramSpecArgs(nil, metal); len(got) != 0 {
			t.Errorf("expected no args for Metal, got %v", got)
		}
	})

	t.Run("no devices", func(t *testing.T) {
		t.Setenv("OLLAMA_VULKAN_NGRAM", "")
		if got := appendVulkanNgramSpecArgs(nil, nil); len(got) != 0 {
			t.Errorf("expected no args with no GPUs, got %v", got)
		}
	})

	t.Run("disabled by env", func(t *testing.T) {
		t.Setenv("OLLAMA_VULKAN_NGRAM", "0")
		if got := appendVulkanNgramSpecArgs(nil, vulkan); len(got) != 0 {
			t.Errorf("OLLAMA_VULKAN_NGRAM=0 should disable it, got %v", got)
		}
	})

	// An explicitly chosen speculative mode -- MTP draft, for instance -- must
	// not be overwritten, and llama-server would reject two --spec-type flags.
	t.Run("yields to an existing spec-type", func(t *testing.T) {
		t.Setenv("OLLAMA_VULKAN_NGRAM", "")
		existing := []string{"--spec-type", "draft-mtp", "--spec-draft-n-max", "4"}
		got := appendVulkanNgramSpecArgs(slices.Clone(existing), vulkan)
		if len(got) != len(existing) {
			t.Errorf("should not have appended to an existing spec-type: %v", got)
		}
		if slices.Contains(got, "ngram-cache") {
			t.Error("overrode an explicitly selected speculative mode")
		}
	})

	t.Run("mixed device set with any vulkan enables it", func(t *testing.T) {
		t.Setenv("OLLAMA_VULKAN_NGRAM", "")
		mixed := []ml.DeviceInfo{
			{DeviceID: ml.DeviceID{ID: "0", Library: "Metal"}},
			{DeviceID: ml.DeviceID{ID: "1", Library: "Vulkan"}},
		}
		if got := appendVulkanNgramSpecArgs(nil, mixed); !slices.Contains(got, "ngram-cache") {
			t.Errorf("expected ngram-cache for a device set containing Vulkan, got %v", got)
		}
	})
}
