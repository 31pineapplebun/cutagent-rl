"""ffprobe JSON parsing contracts independent of human-readable output."""

from pathlib import Path

from cutagent.core.artifacts import ArtifactRef
from cutagent.ingestion.ffprobe import FFprobeAdapter


def _ref(tmp_path: Path, name: str, content: bytes) -> ArtifactRef:
    path = tmp_path / name
    path.write_bytes(content)
    return ArtifactRef.from_path(
        path,
        artifact_id=name.replace(".", "-"),
        media_type="application/json" if name.endswith("json") else "video/mp4",
    )


def test_parser_preserves_rotation_rates_audio_and_nonzero_start(tmp_path: Path) -> None:
    source = _ref(tmp_path, "source.mp4", b"source")
    raw = _ref(tmp_path, "probe.json", b"{}")
    payload = {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "2.500000",
            "start_time": "1.250000",
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "time_base": "1/1000",
                "start_pts": 1250,
                "start_time": "1.250000",
                "duration_ts": 2500,
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24000/1001",
                "r_frame_rate": "24/1",
                "pix_fmt": "yuv420p",
                "nb_frames": "60",
                "side_data_list": [{"rotation": -90}],
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "time_base": "1/48000",
                "start_pts": 60000,
                "start_time": "1.250000",
                "duration_ts": 120000,
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
    }

    video = FFprobeAdapter.parse(payload, source=source, raw_ffprobe=raw)

    assert video.duration_ms == 2500
    assert video.source_start_time_ms == 1250
    assert video.video_stream.source_start_pts == 1250
    assert video.video_stream.rotation_degrees == -90
    assert video.video_stream.variable_frame_rate is True
    assert video.audio_streams[0].channels == 2
