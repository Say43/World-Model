"""nanoWM training infrastructure: model-agnostic trainer, checkpoint/resume,
and DDP setup. Owned by the train-infra subagent. Does not import from
src.model or src.data — it consumes an nn.Module, an optimizer, a loss_fn,
and a torch DataLoader as plain interfaces.
"""
