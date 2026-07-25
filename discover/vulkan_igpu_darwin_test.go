//go:build darwin

package discover

import (
	"runtime"
	"testing"

	"github.com/ollama/ollama/ml"
)

// requireDarwinVulkanPlatform skips tests that exercise filterIntegratedGPUs,
// which short-circuits on Apple Silicon (those Macs use Metal, so the Vulkan
// classification path is unreachable there).
func requireDarwinVulkanPlatform(t *testing.T) {
	t.Helper()
	if runtime.GOARCH == "arm64" {
		t.Skip("filterIntegratedGPUs short-circuits on darwin/arm64")
	}
}

func TestDarwinVulkanDeviceIsIntegrated(t *testing.T) {
	cases := []struct {
		name string
		dev  ml.DeviceInfo
		want bool
	}{
		{
			name: "intel igpu by description",
			dev:  ml.DeviceInfo{Name: "Vulkan1", Description: "Intel(R) UHD Graphics 630"},
			want: true,
		},
		{
			name: "apple igpu by description",
			dev:  ml.DeviceInfo{Name: "Vulkan0", Description: "Apple M1 Pro"},
			want: true,
		},
		{
			name: "vendor carried on name instead of description",
			dev:  ml.DeviceInfo{Name: "Intel(R) Iris(R) Plus Graphics", Description: ""},
			want: true,
		},
		{
			name: "discrete amd stays discrete",
			dev:  ml.DeviceInfo{Name: "Vulkan0", Description: "AMD Radeon Pro 5500M"},
			want: false,
		},
		{
			name: "discrete amd 6000 series stays discrete",
			dev:  ml.DeviceInfo{Name: "Vulkan0", Description: "AMD Radeon Pro 5600M"},
			want: false,
		},
		{
			name: "unknown vendor is not assumed integrated",
			dev:  ml.DeviceInfo{Name: "Vulkan0", Description: "Some Future GPU"},
			want: false,
		},
		{
			name: "case insensitive",
			dev:  ml.DeviceInfo{Name: "Vulkan1", Description: "INTEL(R) UHD GRAPHICS 630"},
			want: true,
		},
		{
			name: "empty device",
			dev:  ml.DeviceInfo{},
			want: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := darwinVulkanDeviceIsIntegrated(tc.dev); got != tc.want {
				t.Errorf("darwinVulkanDeviceIsIntegrated(%q/%q) = %v, want %v",
					tc.dev.Name, tc.dev.Description, got, tc.want)
			}
		})
	}
}

// TestFilterIntegratedGPUsClassifiesDarwinVulkan covers the regression that
// motivated this patch: an unclassified Intel iGPU reporting system RAM as
// VRAM must be dropped so it cannot win device selection or inflate the
// auto-selected context length.
func TestFilterIntegratedGPUsClassifiesDarwinVulkan(t *testing.T) {
	requireDarwinVulkanPlatform(t)
	t.Setenv("OLLAMA_IGPU_ENABLE", "")

	devices := []ml.DeviceInfo{
		{
			DeviceID:    ml.DeviceID{ID: "0", Library: "Vulkan"},
			Name:        "Vulkan0",
			Description: "AMD Radeon Pro 5500M",
			TotalMemory: 8176 * 1024 * 1024,
		},
		{
			DeviceID:    ml.DeviceID{ID: "1", Library: "Vulkan"},
			Name:        "Vulkan1",
			Description: "Intel(R) UHD Graphics 630",
			TotalMemory: 32768 * 1024 * 1024,
		},
	}

	got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))

	if len(got) != 1 {
		t.Fatalf("expected 1 device after filtering, got %d: %+v", len(got), got)
	}
	if got[0].Description != "AMD Radeon Pro 5500M" {
		t.Errorf("wrong device survived filtering: %q", got[0].Description)
	}
}

// TestFilterIntegratedGPUsLeavesNonVulkanAlone guards against the darwin
// classifier reaching beyond the Vulkan backend it is scoped to.
func TestFilterIntegratedGPUsLeavesNonVulkanAlone(t *testing.T) {
	requireDarwinVulkanPlatform(t)
	devices := []ml.DeviceInfo{
		{
			DeviceID:    ml.DeviceID{ID: "0", Library: "Metal"},
			Name:        "Metal0",
			Description: "Apple M1 Pro",
			TotalMemory: 16384 * 1024 * 1024,
		},
	}

	got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))

	if len(got) != 1 {
		t.Fatalf("Metal device should not be classified by the Vulkan path, got %d devices", len(got))
	}
}
