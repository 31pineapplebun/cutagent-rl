#!/usr/bin/env python3
"""Probe the real machine environment without inferring unavailable capabilities."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
MIB = 1024 * 1024
UTC = timezone.utc  # noqa: UP017 -- standalone probe must run on the server's Python 3.10


def _run(command: Sequence[str], timeout_seconds: int = 10) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "executable": None,
            "return_code": None,
            "stdout": "",
            "stderr": "not found on PATH",
        }
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": True,
            "executable": executable,
            "return_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "executable": executable,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _command_text(result: dict[str, object], key: str = "stdout") -> str:
    value = result.get(key)
    return value if isinstance(value, str) else ""


def _distribution() -> dict[str, str | None]:
    try:
        release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        release = {}
    return {
        "id": release.get("ID"),
        "name": release.get("NAME"),
        "version_id": release.get("VERSION_ID"),
        "pretty_name": release.get("PRETTY_NAME"),
    }


def _parse_cpu_model(cpuinfo: str) -> str | None:
    candidates: dict[str, list[str]] = {"model name": [], "hardware": [], "processor": []}
    for line in cpuinfo.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().casefold()
        candidate = value.strip()
        if normalized_key in candidates and candidate:
            candidates[normalized_key].append(candidate)
    for key in ("model name", "hardware", "processor"):
        for candidate in candidates[key]:
            if key != "processor" or not candidate.isdigit():
                return candidate
    return None


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        detected = _parse_cpu_model(cpuinfo.read_text(encoding="utf-8", errors="replace"))
        if detected:
            return detected
    processor = platform.processor().strip()
    if processor:
        return processor
    environment_processor = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    return environment_processor or None


def _memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024

    if os.name == "nt":
        powershell = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        value = _command_text(powershell)
        if powershell.get("return_code") == 0 and value.isdigit():
            return int(value)

    sysconf = getattr(os, "sysconf", None)
    if callable(sysconf):
        try:
            pages = int(sysconf("SC_PHYS_PAGES"))
            page_size = int(sysconf("SC_PAGE_SIZE"))
        except (OSError, TypeError, ValueError):
            return None
        return pages * page_size
    return None


def _nvidia_probe() -> dict[str, object]:
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, object]] = []
    driver_versions: set[str] = set()
    if query.get("return_code") == 0:
        for line in _command_text(query).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                continue
            index, name, memory_mib, driver, uuid = parts
            try:
                memory_bytes: int | None = int(float(memory_mib)) * MIB
            except ValueError:
                memory_bytes = None
            devices.append(
                {
                    "index": int(index) if index.isdigit() else index,
                    "name": name,
                    "memory_total_bytes": memory_bytes,
                    "uuid": uuid,
                }
            )
            if driver:
                driver_versions.add(driver)

    full = _run(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", _command_text(full))
    return {
        "nvidia_smi_available": bool(query.get("available")),
        "nvidia_smi_executable": query.get("executable"),
        "query_succeeded": query.get("return_code") == 0,
        "devices": devices,
        "driver_versions": sorted(driver_versions),
        "driver_supported_cuda_version": cuda_match.group(1) if cuda_match else None,
        "probe_error": _command_text(query, "stderr") or None,
    }


def _cuda_toolkit_probe() -> dict[str, object]:
    result = _run(["nvcc", "--version"])
    output = "\n".join(filter(None, (_command_text(result), _command_text(result, "stderr"))))
    match = re.search(r"release\s+([0-9.]+)", output)
    return {
        "nvcc_available": bool(result.get("available")),
        "nvcc_executable": result.get("executable"),
        "toolkit_version": match.group(1) if match else None,
        "raw_version": output or None,
    }


def _pytorch_probe() -> dict[str, object]:
    if importlib.util.find_spec("torch") is None:
        return {
            "installed": False,
            "version": None,
            "compiled_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cudnn_version": None,
            "error": None,
        }
    try:
        torch: Any = importlib.import_module("torch")
        cuda: Any = torch.cuda
        backends: Any = torch.backends
        version_info: Any = torch.version
        return {
            "installed": True,
            "version": str(torch.__version__),
            "compiled_cuda_version": getattr(version_info, "cuda", None),
            "cuda_available": bool(cuda.is_available()),
            "cuda_device_count": int(cuda.device_count()),
            "cudnn_version": backends.cudnn.version(),
            "error": None,
        }
    except Exception as exc:  # importing a mismatched binary can raise many exception types
        return {
            "installed": True,
            "version": None,
            "compiled_cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cudnn_version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _media_binary_probe(name: str) -> dict[str, object]:
    result = _run([name, "-version"])
    if not result.get("available"):
        return {
            "available": False,
            "executable": None,
            "return_code": None,
            "version_line": None,
            "error": _command_text(result, "stderr") or None,
        }
    output = _command_text(result) or _command_text(result, "stderr")
    first_line = output.splitlines()[0] if output else None
    return {
        "available": bool(result.get("available")),
        "executable": result.get("executable"),
        "return_code": result.get("return_code"),
        "version_line": first_line,
        "error": _command_text(result, "stderr") or None,
    }


def _docker_probe() -> dict[str, object]:
    version = _run(["docker", "--version"])
    if not version.get("available"):
        return {
            "cli_available": False,
            "executable": None,
            "version": None,
            "daemon_accessible": False,
            "daemon_error": None,
        }
    info = _run(["docker", "info", "--format", "{{json .}}"], timeout_seconds=15)
    return {
        "cli_available": True,
        "executable": version.get("executable"),
        "version": _command_text(version) or None,
        "daemon_accessible": info.get("return_code") == 0,
        "daemon_error": _command_text(info, "stderr") or None,
    }


def _version_command(command: Sequence[str]) -> dict[str, object]:
    result = _run(command)
    output = _command_text(result) or _command_text(result, "stderr")
    return {
        "available": bool(result.get("available")),
        "executable": result.get("executable"),
        "return_code": result.get("return_code"),
        "version": output.splitlines()[0] if output else None,
        "error": (
            _command_text(result, "stderr") or None
            if result.get("return_code") not in (0, None)
            else None
        ),
    }


def _package_manager_probe() -> dict[str, object]:
    pip_result = _run([sys.executable, "-m", "pip", "--version"])
    pip_output = _command_text(pip_result) or _command_text(pip_result, "stderr")
    return {
        "uv": _version_command(["uv", "--version"]),
        "pip": {
            "available": pip_result.get("return_code") == 0,
            "executable": sys.executable,
            "version": pip_output.splitlines()[0] if pip_output else None,
            "error": (
                _command_text(pip_result, "stderr") or None
                if pip_result.get("return_code") != 0
                else None
            ),
        },
        "conda": _version_command(["conda", "--version"]),
        "apt": _version_command(["apt-get", "--version"]),
        "dpkg": _version_command(["dpkg-query", "--version"]),
    }


def _container_probe() -> dict[str, object]:
    indicators: list[str] = []
    if Path("/.dockerenv").exists():
        indicators.append("/.dockerenv")
    cgroup = Path("/proc/1/cgroup")
    if cgroup.is_file():
        content = cgroup.read_text(encoding="utf-8", errors="replace").casefold()
        for runtime in ("docker", "containerd", "kubepods", "podman", "lxc"):
            if runtime in content:
                indicators.append(f"cgroup:{runtime}")
    return {"detected": bool(indicators), "indicators": indicators}


def collect_environment() -> dict[str, object]:
    """Collect a machine-readable manifest from observed local state."""

    uname = platform.uname()
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "os": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "kernel": platform.release(),
            "distribution": _distribution(),
        },
        "cpu": {
            "architecture": platform.machine() or None,
            "model": _cpu_model(),
            "logical_count": os.cpu_count(),
        },
        "memory": {"total_bytes": _memory_total_bytes()},
        "gpu": _nvidia_probe(),
        "cuda_toolkit": _cuda_toolkit_probe(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "virtual_environment": sys.prefix != sys.base_prefix,
        },
        "pytorch": _pytorch_probe(),
        "ffmpeg": _media_binary_probe("ffmpeg"),
        "ffprobe": _media_binary_probe("ffprobe"),
        "docker": _docker_probe(),
        "container": _container_probe(),
        "package_managers": _package_manager_probe(),
    }


def write_manifest(manifest: dict[str, object], output: str, *, pretty: bool = True) -> None:
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=True,
    )
    if output == "-":
        print(serialized)
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(f"{serialized}\n", encoding="utf-8")
    temporary.replace(destination)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/environment/environment.json",
        help="manifest path, or '-' for stdout",
    )
    parser.add_argument("--compact", action="store_true", help="write compact JSON")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    manifest = collect_environment()
    write_manifest(manifest, args.output, pretty=not args.compact)
    if args.output != "-":
        print(f"environment manifest written to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
