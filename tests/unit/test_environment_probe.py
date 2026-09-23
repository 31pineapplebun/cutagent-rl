"""Standalone environment probe contract tests."""

import ast
import json
from pathlib import Path

from scripts.gpu_smoke import _base_result
from scripts.probe_environment import _parse_cpu_model, write_manifest


def test_probe_remains_compatible_with_python_310() -> None:
    source = Path("scripts/probe_environment.py").read_text(encoding="utf-8")
    ast.parse(source, filename="probe_environment.py", feature_version=(3, 10))
    assert "from datetime import UTC" not in source


def test_environment_manifest_is_written_as_machine_readable_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "environment.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "os": {"system": "observed-test-value"},
    }
    write_manifest(manifest, str(output))

    assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_linux_cpu_parser_does_not_treat_processor_index_as_model() -> None:
    cpuinfo = """processor : 0
vendor_id : GenuineIntel
model name : Example Research CPU 9000
processor : 1
model name : Example Research CPU 9000
"""
    assert _parse_cpu_model(cpuinfo) == "Example Research CPU 9000"


def test_gpu_smoke_result_starts_failed_until_real_checks_pass() -> None:
    result = _base_result(7, "Example GPU")
    assert result["status"] == "failed"
    assert result["seed"] == 7
    assert result["expected_gpu"] == "Example GPU"
