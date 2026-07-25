//go:build darwin

package discover

import (
	"strings"

	"github.com/ollama/ollama/ml"
)

// integratedVulkanVendorsDarwin lists the GPU vendors that only ever ship
// integrated parts in Macs. Every Intel and Apple GPU in a Mac is on-package;
// only AMD (and, on very old machines, NVIDIA) appear as discrete devices.
var integratedVulkanVendorsDarwin = []string{"intel", "apple"}

// darwinVulkanDeviceIsIntegrated reports whether a Vulkan device should be
// treated as an integrated GPU on macOS.
//
// Ollama normally learns this from the "ggml_vulkan: ... uma: 1" metadata
// llama-server prints during discovery, but MoltenVK-backed builds do not emit
// those lines, so iGPUs arrive unclassified. An unclassified Intel iGPU is
// reported with the machine's full system RAM as its VRAM (32 GiB on a 16 GiB
// Mac), which inflates the auto-selected context length and makes the
// scheduler prefer the wrong device.
//
// The Vulkan loader does not expose a vendor ID through this code path, and
// PCIID is empty under MoltenVK, so vendor identification falls back to the
// backend-reported strings. Both Name and Description are checked because
// which one carries the human-readable vendor differs between MoltenVK
// versions.
func darwinVulkanDeviceIsIntegrated(dev ml.DeviceInfo) bool {
	haystack := strings.ToLower(dev.Description + " " + dev.Name)
	for _, vendor := range integratedVulkanVendorsDarwin {
		if strings.Contains(haystack, vendor) {
			return true
		}
	}
	return false
}
