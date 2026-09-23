# M1A media timebase contract

CutAgent exposes temporal coordinates as integer milliseconds on a normalized,
zero-based timeline. Every interval is half-open: `[start_ms, end_ms)`. The
source file and its original stream timing metadata remain immutable.

## Source PTS mapping

For a selected video stream with time base `num/den` and source origin PTS
`origin_pts`, a decoded source timestamp maps to normalized time as:

```text
delta_ticks = pts - origin_pts
normalized_ms = round(delta_ticks * num * 1000 / den)
```

Point timestamps use nearest rounding with exact half values rounded away from
zero. Interval starts use floor and interval ends use ceil. The asymmetric
interval rule preserves source coverage at millisecond boundaries.

The inverse mapping is:

```text
source_pts = origin_pts + round(normalized_ms * den / (num * 1000))
```

All calculations use integer/Fraction/Decimal arithmetic. Binary floating-point
frame-rate calculations are not used. In particular, `frame_index / fps` is
never accepted as canonical temporal evidence.

## CFR, VFR and non-zero starts

- CFR average/real rates are descriptive stream metadata, not the clock.
- VFR frames retain their decoded PTS spacing and use the same PTS conversion.
- A non-zero or negative stream start remains in `source_start_pts` and
  `source_start_time_ms`; subtracting that origin maps the first source instant
  to normalized `0 ms`.
- Segment identity and keyframe requests use normalized timestamps, not frame
  indexes.

## Analysis proxies

The immutable source is used directly by default. Under explicit `auto`
normalization, an analysis proxy is created only for a declared incompatibility
such as unsupported codec/pixel format or excessive width. FFmpeg uses timestamp
copying and passthrough VSync; it does not force CFR.

The M1A proxy is visual-only. Audio stays on the immutable source because audio
encoder priming can shift container duration and because later ASR must use the
source audio clock. This omission is explicit in the transformation config.

Every proxy records both source and proxy ArtifactRefs, the transformation
configuration, FFmpeg version, source/proxy time bases and origins, and the rule
`subtract_origin_then_scale_to_ms`. Source and proxy therefore meet on the same
normalized millisecond timeline. Outputs intended for editing or evaluation map
back through the immutable source origin/time base.

## Downstream contract

ASR, OCR, VLM observations, retrieval, FFmpeg editing tools, and benchmark
temporal IoU must exchange `TimeRange` values on the normalized timeline. When
they retain decoded evidence, they should also retain the corresponding source
PTS and time base. Intervals beyond a known duration are rejected; explicit
clipping must be requested and cannot yield an empty interval.
