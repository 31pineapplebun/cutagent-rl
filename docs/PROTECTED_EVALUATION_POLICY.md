# Protected Evaluation Policy

CutAgentBench v0.1 `locked_test` and `adversarial_test` are sealed. Until two validated independent
real-human calibration submissions and required adjudication produce a frozen gate hash:

- no Agent/model execution on protected TaskInputs;
- no ordinary deserialization of protected Gold;
- no per-task protected errors or tuning;
- protected access count remains zero.

Schema, task-ID, seal, and hash verification are allowed without Gold deserialization. After the
gate, each preregistered selected model may run once with an access record containing Git commit,
model revision, adapter/checkpoint hash, config hash, benchmark version, reason, declared metrics,
timestamp, and output path. Protected results never feed retraining, prompt changes, threshold
selection, or benchmark changes.
