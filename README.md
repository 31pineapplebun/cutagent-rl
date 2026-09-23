# CutAgent-RL

CutAgent-RL is a research prototype for multimodal video-agent workflows: video
ingestion, perception and retrieval, FFmpeg editing tools, a typed Agent runtime,
benchmark evaluation, and small SFT/DPO/reward-model/GRPO experiments.

This repository is a **curated public source snapshot**, not a mirror of the
original private research repository. Its earlier commits, operational notes,
machine-specific configuration, model weights, media, raw data, protected Gold,
and experiment artifacts are not part of this snapshot. The code does not
silently substitute mock inference for a requested real model run.

## Research status

The post-training experiments did not demonstrate an improvement in full-Agent
task success. In the completed protected comparison, prompt-only and M6 SFT
both scored 0/60 on the locked split; on the 30-task adversarial split they
scored 4/30 and 0/30, respectively. The four prompt-only successes were correct
refusals of impossible tasks, not successful video edits.

A separate 20-task **development** regression on one generated source compared
baseline handoff with compact recovery. Ordinary edits were 10/12 versus 11/12,
one-shot timeout cases 0/6 versus 1/6, and correct refusals 2/2 for both. These
small, overlapping cases do not establish generalization or production
reliability. The private evaluation artifacts are not distributed in this
public snapshot; these figures are reported historical observations, not a
fresh public rerun. See [the publication report](reports/PUBLICATION_REPORT.md).

## Local development

Python 3.11+, `uv`, and FFmpeg/FFprobe are needed for the full local test suite.
From the repository root:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy src evaluation/src training/src scripts
```

Ordinary unit tests do not download large models. Real model inference needs
the appropriate optional dependencies, licensed input media, and a model cache
supplied by the user. The reduced Agent entry is `python -m scripts.harness_run
--help`; it does not perform fresh perception for arbitrary new videos.
Protected evaluation cannot be reconstructed from this snapshot alone because
the sealed material, calibration submissions, and run artifacts are not public.

The [media timebase contract](docs/MEDIA_TIMEBASE.md),
[protected-evaluation policy](docs/PROTECTED_EVALUATION_POLICY.md), and
[reward/GRPO protocol](docs/REWARD_AND_GRPO_PROTOCOL.md) describe selected
research interfaces without private operational material.

The current [LICENSE](LICENSE) reserves all rights. Public visibility does not
grant permission to reuse the software or redistribute associated models/media.
