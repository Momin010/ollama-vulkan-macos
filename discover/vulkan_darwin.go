//go:build darwin

package discover

import (
	"os"
	"path/filepath"

	"github.com/ollama/ollama/ml"
)

// Vulkan on macOS runs through MoltenVK, a translation layer over Metal. Two
// environment defaults are required for it to work correctly, and both are set
// here rather than documented as manual steps because the desktop app launches
// the server with a minimal environment in which the user has no opportunity
// to set them. Runner subprocesses inherit these via the server's environment.
//
// Both are only defaults: an explicitly set value is always left alone.
func init() {
	// MoltenVK's fp16 matmul path aborts the device with
	// vk::ErrorDeviceLost once prompt processing runs a large batch
	// (reproduced on an AMD Radeon Pro 5500M at batch 512, roughly 1-2k
	// tokens of prompt). The f32 path is stable, and on this hardware it is
	// also substantially faster at prefill than the alternative workaround
	// of shrinking the batch size.
	setDefaultEnv("GGML_VK_DISABLE_F16", "1")

	if icd := findMoltenVKICD(); icd != "" {
		setDefaultEnv("VK_ICD_FILENAMES", icd)
	}
}

// setDefaultEnv sets key to value only if it is not already set, so every
// decision made here can be overridden from the environment.
func setDefaultEnv(key, value string) {
	if os.Getenv(key) == "" {
		os.Setenv(key, value)
	}
}

// moltenVKICDCandidates returns the paths to search for a MoltenVK ICD
// manifest, in priority order.
//
// The bundled copy comes first: release builds ship MoltenVK and the Vulkan
// loader inside the runner payload so the target machine needs neither
// Homebrew nor the Vulkan SDK, and a bundled ICD is guaranteed to match the
// loader it was built against. Homebrew and Vulkan SDK locations follow, for
// builds from source.
func moltenVKICDCandidates() []string {
	candidates := []string{
		filepath.Join(ml.LibOllamaPath, "vulkan", "MoltenVK_icd.json"),
	}

	if sdk := os.Getenv("VULKAN_SDK"); sdk != "" {
		candidates = append(candidates,
			filepath.Join(sdk, "share", "vulkan", "icd.d", "MoltenVK_icd.json"),
			filepath.Join(sdk, "etc", "vulkan", "icd.d", "MoltenVK_icd.json"),
		)
	}

	return append(candidates,
		"/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json",    // Homebrew (Intel)
		"/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json", // Homebrew (Apple Silicon)
		"/usr/local/share/vulkan/icd.d/MoltenVK_icd.json",  // Vulkan SDK, system install
	)
}

// findMoltenVKICD returns the first MoltenVK ICD manifest that exists on disk,
// or "" if none is found. An empty result is not fatal: the Vulkan loader has
// its own search paths, and a machine with no MoltenVK at all reports no
// Vulkan devices and falls back to whatever backend is available.
func findMoltenVKICD() string {
	for _, candidate := range moltenVKICDCandidates() {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	return ""
}
