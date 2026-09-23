# Reward Model and GRPO Protocol

Reward Model v1 has only pairwise scalar/ranking reward and compact failure classification.
Replan/termination heads and dimension regression are out of scope. RM inputs obey the public
visibility boundary; labels are stored and loaded separately. Pairwise/failure losses and metrics
are logged independently, including calibration, length correlation, and shortcut diagnostics.

GRPO starts on short-horizon objectively verifiable train tasks. Four candidates share one
initial environment snapshot. Rewards log rule progress, valid capability/arguments, observable
recovery, termination, optional low-weight RM score, malformed/invalid/repeated/unsafe penalties,
total environment reward, and the separate KL objective. Curriculum is single decision, one tool,
two tools, then observable recovery only after stability. Validation never supplies rewards.

High-reward anomalies, empty/repetitive outputs, parser-default exploitation, fabricated evidence,
premature termination, RM shortcuts, and length/reward correlation are mandatory audit targets.
