# Public source snapshot report

## Goal and scope

Publish a useful source-only view of CutAgent-RL while keeping the original
research history and operational evidence private. This is a curated snapshot,
not a replacement for the private experiment archive.

## What was implemented

- Included Agent, evaluation, training, tool, and test source code, dependency
  manifests, and selected protocol documentation.
- Replaced machine-specific defaults in public smoke/finalization scripts with
  generic behavior or an explicit local-checkpoint placeholder.
- Excluded historical reports, operational runbooks, media, model weights,
  private data, protected Gold, and generated experiment outputs.

## Architecture and design decisions

The public repository begins with a new root commit. This prevents the private
repository's prior commits and tags from becoming part of its Git history.
The private archive remains separate. The public snapshot retains modular
runtime boundaries and typed Agent/evaluation contracts, but does not claim
that excluded private artifacts can be reproduced from source alone.

## Files changed

The snapshot was assembled from the private source tree. Public-specific changes
are `README.md`, this report, generic GPU smoke behavior, the local checkpoint
placeholder in `scripts/finalize_after_human_gate.py`, and synthetic test
fixtures. Private operational documents and historical reports were omitted.

## Dependencies changed

None. The source and dependency manifests retain their existing versions.

## Tests and quality gates

- Unit/integration suite: **271 passed, 5 skipped**. Three skips require
  private calibration artifacts intentionally omitted from this snapshot; one
  requires unavailable symlink creation, and one requires Pillow in the local
  test environment.
- Ruff: passed with Git-ignore handling disabled for the staging directory.
- mypy: passed for 187 source files in `src`, `evaluation/src`,
  `training/src`, and `scripts`.
- M0 synthetic infrastructure smoke: passed with identical state replay.
- M1A real FFmpeg/FFprobe ingestion smoke: passed with three scenes and three
  keyframes on generated, non-private media.
- Gitleaks scans of the staged files and public Git commit: no leaks found.
  A later whole-directory scan flagged 17 generic-key patterns in generated,
  Git-ignored smoke outputs; none of those files were staged or published.
  Additional checks of committed content found no private network endpoint,
  machine path, personal email, or concrete device model. Pattern-based checks
  cannot prove the absence of every possible undisclosed secret.
- The initial public commit was scanned again as Git history: one new root
  commit, no historical tags, and no Gitleaks findings. Its author and
  committer use a GitHub-provided `noreply` address.
- After publication, anonymous access to the public repository and the new
  commit succeeded. An old private-history commit was not accessible at the
  public repository URL. The separate original archive remained private.

## Real experiment results

No new model or GPU experiment was run for this publication. The README reports
historical aggregate observations from the private research archive and clearly
states that the underlying artifacts are not included here.

## Produced artifacts

The publication artifact is
[the public Git repository](https://github.com/31pineapplebun/cutagent-rl).
No model weights, videos, private evaluation files, or machine-specific
artifacts are published.

## Known limitations

The public snapshot is not an independently complete protected-evaluation
reproduction package. Its historical aggregate results cannot be verified from
the public source alone. Model inference requires separately obtained models,
dependencies, and authorized media.

## Unresolved or blocking issues

No known publication blocker remains in this curated snapshot. Future changes
to the public repository should pass the same staged-file and history checks.

## Recommended next milestone

If a broader reproducibility package is wanted, prepare a separately reviewed,
rights-cleared public development-only dataset and evidence bundle. Do not
publish sealed evaluation material or the private operational archive.
