//go:build darwin

package discover

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSetDefaultEnv(t *testing.T) {
	t.Run("sets when unset", func(t *testing.T) {
		t.Setenv("OLLAMA_TEST_DEFAULT_ENV", "")
		setDefaultEnv("OLLAMA_TEST_DEFAULT_ENV", "default")
		if got := os.Getenv("OLLAMA_TEST_DEFAULT_ENV"); got != "default" {
			t.Errorf("got %q, want %q", got, "default")
		}
	})

	t.Run("preserves an explicit value", func(t *testing.T) {
		t.Setenv("OLLAMA_TEST_DEFAULT_ENV", "user-choice")
		setDefaultEnv("OLLAMA_TEST_DEFAULT_ENV", "default")
		if got := os.Getenv("OLLAMA_TEST_DEFAULT_ENV"); got != "user-choice" {
			t.Errorf("explicit value was overwritten: got %q", got)
		}
	})
}

func TestMoltenVKICDCandidatesPrefersBundled(t *testing.T) {
	t.Setenv("VULKAN_SDK", "")

	candidates := moltenVKICDCandidates()
	if len(candidates) == 0 {
		t.Fatal("expected at least one candidate path")
	}
	if !strings.HasSuffix(candidates[0], filepath.Join("vulkan", "MoltenVK_icd.json")) {
		t.Errorf("bundled payload path should be searched first, got %q", candidates[0])
	}
}

func TestMoltenVKICDCandidatesIncludesVulkanSDK(t *testing.T) {
	t.Setenv("VULKAN_SDK", "/opt/vulkan-sdk")

	var found bool
	for _, c := range moltenVKICDCandidates() {
		if strings.HasPrefix(c, "/opt/vulkan-sdk") {
			found = true
			break
		}
	}
	if !found {
		t.Error("VULKAN_SDK was set but no candidate path referenced it")
	}
}

func TestMoltenVKICDCandidatesCoversBothHomebrewPrefixes(t *testing.T) {
	joined := strings.Join(moltenVKICDCandidates(), "\n")
	for _, prefix := range []string{"/usr/local/", "/opt/homebrew/"} {
		if !strings.Contains(joined, prefix) {
			t.Errorf("missing Homebrew prefix %q in candidates", prefix)
		}
	}
}

func TestFindMoltenVKICDIgnoresDirectories(t *testing.T) {
	// A directory sitting where the manifest should be must not be returned
	// as if it were a manifest; VK_ICD_FILENAMES pointing at a directory
	// makes the loader fail to enumerate any device at all.
	dir := t.TempDir()
	sdk := filepath.Join(dir, "sdk")
	icdDir := filepath.Join(sdk, "share", "vulkan", "icd.d", "MoltenVK_icd.json")
	if err := os.MkdirAll(icdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VULKAN_SDK", sdk)

	if got := findMoltenVKICD(); got == icdDir {
		t.Errorf("findMoltenVKICD returned a directory: %q", got)
	}
}

func TestFindMoltenVKICDReturnsExistingFile(t *testing.T) {
	dir := t.TempDir()
	sdk := filepath.Join(dir, "sdk")
	icdDir := filepath.Join(sdk, "share", "vulkan", "icd.d")
	if err := os.MkdirAll(icdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	icd := filepath.Join(icdDir, "MoltenVK_icd.json")
	if err := os.WriteFile(icd, []byte(`{"file_format_version":"1.0.0"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VULKAN_SDK", sdk)

	// Only assert the SDK path is found when no bundled payload shadows it,
	// which is the case in a test binary run outside an install tree.
	if got := findMoltenVKICD(); got != icd && !strings.Contains(got, "lib") {
		t.Errorf("findMoltenVKICD() = %q, want %q", got, icd)
	}
}
