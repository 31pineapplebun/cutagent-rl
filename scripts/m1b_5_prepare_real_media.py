"""Download, verify, clip, and record the frozen M1B.5 licensed real-video set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from cutagent_evaluation.licensed_media import (
    DerivedClipProvenance,
    LicensedMediaManifest,
    LicensedMediaProvenance,
    SourceMediaProvenance,
)

CHUNK_SIZE = 8 * 1024 * 1024
USER_AGENT = "CutAgent-RL-M1B5-license-validation/1.0 (research evaluation)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{completed.stderr}")
    return completed


def _version(executable: str) -> str:
    first_line = _run([executable, "-version"]).stdout.splitlines()[0].strip()
    if not first_line:
        raise RuntimeError(f"{executable} returned an empty version")
    return first_line


def _download(url: str, destination: Path) -> tuple[str, bool]:
    if destination.is_file() and destination.stat().st_size > 0:
        return url, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resolved = response.geturl()
                with partial.open("wb") as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
            partial.replace(destination)
            return resolved, False
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if partial.exists():
                partial.unlink()
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _duration_ms(ffprobe: str, path: Path) -> int:
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    duration = json.loads(completed.stdout)["format"]["duration"]
    return round(float(duration) * 1000)


def _extract_clip(
    ffmpeg: str, source: Path, destination: Path, start_ms: int, end_ms: int
) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-t",
        f"{(end_ms - start_ms) / 1000:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        str(destination),
    ]
    if not destination.is_file() or destination.stat().st_size == 0:
        _run(command)
    return command


def _contact_sheet(ffmpeg: str, clip: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0:
        return
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clip),
            "-vf",
            "fps=1,scale=320:-2,tile=3x1",
            "-frames:v",
            "1",
            str(output),
        ]
    )


def _previous_sources(output: Path) -> dict[str, SourceMediaProvenance]:
    if not output.is_file():
        return {}
    try:
        previous = LicensedMediaProvenance.model_validate_json(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {source.source_id: source for source in previous.sources}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "evaluation/data/m1b_5_licensed_sources.json",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=repository_root / "artifacts/m1b_5/real_media",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = LicensedMediaManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    output_path = work_root / "provenance.json"
    prior_sources = _previous_sources(output_path)
    now = datetime.now(UTC)
    source_records: list[SourceMediaProvenance] = []
    for source_index, source in enumerate(manifest.sources):
        suffix = Path(source.title).suffix.casefold() or ".video"
        source_path = work_root / "sources" / f"{source.source_id}{suffix}"
        resolved_url, source_hit = _download(source.download_url, source_path)
        prior_source = prior_sources.get(source.source_id)
        if source_hit and prior_source is not None:
            resolved_url = prior_source.resolved_download_url
        license_path = work_root / "license_pages" / f"{source.source_id}.html"
        _, license_hit = _download(source.source_page, license_path)
        source_duration_ms = _duration_ms(args.ffprobe, source_path)
        clips: list[DerivedClipProvenance] = []
        for clip in source.clips:
            if clip.end_ms > source_duration_ms + 100:
                raise RuntimeError(
                    f"clip {clip.clip_id} ends after source duration: "
                    f"{clip.end_ms}>{source_duration_ms}"
                )
            clip_path = work_root / "clips" / f"{clip.clip_id}.mp4"
            command = _extract_clip(
                args.ffmpeg,
                source_path,
                clip_path,
                clip.start_ms,
                clip.end_ms,
            )
            _contact_sheet(
                args.ffmpeg,
                clip_path,
                work_root / "contact_sheets" / f"{clip.clip_id}.jpg",
            )
            clips.append(
                DerivedClipProvenance(
                    clip_id=clip.clip_id,
                    source_id=source.source_id,
                    start_ms=clip.start_ms,
                    end_ms=clip.end_ms,
                    sha256=_sha256(clip_path),
                    size_bytes=clip_path.stat().st_size,
                    relative_path=clip_path.relative_to(work_root).as_posix(),
                    extraction_command=tuple(command),
                )
            )
        source_records.append(
            SourceMediaProvenance(
                source_id=source.source_id,
                title=source.title,
                source_page=source.source_page,
                resolved_download_url=resolved_url,
                license=source.license,
                license_url=source.license_url,
                attribution=source.attribution,
                license_page_sha256=_sha256(license_path),
                license_page_relative_path=license_path.relative_to(work_root).as_posix(),
                download_date_utc=(
                    prior_source.download_date_utc if prior_source is not None else now
                ),
                sha256=_sha256(source_path),
                size_bytes=source_path.stat().st_size,
                duration_ms=source_duration_ms,
                relative_path=source_path.relative_to(work_root).as_posix(),
                clips=tuple(clips),
            )
        )
        print(
            f"prepared source={source.source_id} clips={len(clips)} "
            f"source_cache_hit={source_hit} license_cache_hit={license_hit}"
        )
        if source_index + 1 < len(manifest.sources):
            time.sleep(2)
    provenance = LicensedMediaProvenance(
        dataset_id=manifest.dataset_id,
        manifest_sha256=_sha256(manifest_path),
        prepared_at_utc=now,
        ffmpeg_version=_version(args.ffmpeg),
        ffprobe_version=_version(args.ffprobe),
        sources=tuple(source_records),
    )
    output_path.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"M1B.5 licensed media prepared: sources={len(provenance.sources)} "
        f"clips={provenance.clip_count} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
