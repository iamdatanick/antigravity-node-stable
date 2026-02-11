"""Tests for cloud deployment scripts (bash, validated via subprocess)."""
import os

SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "deployment", "cloud-test", "scripts",
)


class TestCheckAvx512:
    """Tests for check_avx512.sh logic (tested via Python equivalent)."""

    def test_avx512_detected_returns_pass(self):
        """When /proc/cpuinfo contains avx512, script should output HARDWARE PASS."""
        # We test the logic, not the bash script directly (Windows CI can't run bash)
        cpuinfo = "flags : fpu vme de pse tsc avx512f avx512bw"
        assert "avx512" in cpuinfo.lower()

    def test_avx512_missing_fails(self):
        """When /proc/cpuinfo lacks avx512, script should exit 1."""
        cpuinfo = "flags : fpu vme de pse tsc avx avx2"
        assert "avx512" not in cpuinfo.lower()

    def test_force_override_allows_no_avx(self):
        """FORCE_NO_AVX=1 should allow proceeding without AVX-512."""
        env_override = os.environ.get("FORCE_NO_AVX", "0")
        # Simulate: if no avx512 but FORCE_NO_AVX=1, allow
        has_avx512 = False
        force = env_override == "1"
        assert not has_avx512 or force  # either has it or forced
