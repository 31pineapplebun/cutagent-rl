"""The deployable cutagent package cannot import evaluator-private code."""

import ast
import tomllib
from pathlib import Path

import cutagent


def test_runtime_source_does_not_import_private_evaluation_package() -> None:
    runtime_root = Path("src/cutagent")
    violations: list[str] = []
    for path in runtime_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith("cutagent_evaluation") for name in names):
                violations.append(str(path))
    assert violations == []


def test_runtime_wheel_configuration_excludes_evaluation_package() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert packages == ["src/cutagent"]
    assert "cutagent_evaluation" not in cutagent.__all__
    assert "BenchmarkGold" not in cutagent.__all__
