"""Exponential moving average of model weights, kept in fp32 on CPU.

Per the project's hard constraints (fp16 AMP training on T4), the EMA copy
is deliberately decoupled from the training dtype/device: it always lives
as fp32 tensors on CPU, regardless of what device/dtype the live model is
training in. This avoids fp16 accumulation error compounding over tens of
thousands of EMA updates, and keeps EMA memory off the GPU entirely.
"""
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: str = "cpu"):
        # fp32 always; the device is a per-run choice. CPU is the default
        # (the hard-constraint text above). M4's 40M model showed why the
        # alternative exists: copying 162 MB of weights over PCIe every
        # step and averaging them on the CPU is a visible share of the step
        # time at that size, while 162 MB of fp32 shadow on a 15 GB T4 is
        # nothing. Checkpoints always store the shadow on CPU either way.
        self.decay = decay
        self.device = torch.device(device)
        self.shadow = {
            k: v.detach().float().to(self.device).clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            v_dev = v.detach().float().to(self.device)
            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v_dev, alpha=1.0 - self.decay)
            else:
                # Non-floating buffers (e.g. counters, bool masks) aren't
                # averaged -- just kept in sync with the live model.
                self.shadow[k] = v_dev.clone()

    def copy_to(self, model: torch.nn.Module) -> None:
        """Load the EMA weights into `model` in place (e.g. for eval)."""
        msd = model.state_dict()
        for k in msd:
            msd[k].copy_(self.shadow[k].to(dtype=msd[k].dtype, device=msd[k].device))

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.to(self.device).clone() for k, v in state.items()}
