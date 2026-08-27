"""Checkpoint/resume that survives a SIGTERM at 11:45h into a 12h Kaggle session.

Two properties are load-bearing here:

1. Atomic writes. A checkpoint is first written to a temp file in the same
   directory (so it's on the same filesystem) and then moved into place
   with os.replace(), which is atomic on both POSIX and Windows. If the
   process is killed mid-write, the temp file is corrupt but the
   previously-committed checkpoint (and its `latest.json` pointer) is
   untouched -- a partial write can never look like a valid checkpoint.

2. SIGTERM handling that still lets the caller log to the budget ledger.
   Kaggle sends SIGTERM before the hard kill; we must (a) write a full,
   atomic checkpoint synchronously in the handler, then (b) unwind back to
   `scripts/train.py`'s try/finally so it still calls `budget.py log`. We
   do this by having the handler raise SigtermInterrupt instead of calling
   os._exit() -- Python delivers the signal on the main thread between
   bytecode instructions, so the raise propagates up through the normal
   call stack (through Trainer.fit) exactly like any other exception.

Platform note: real OS-level SIGTERM delivery (`os.kill(pid, SIGTERM)`)
is POSIX behavior (what Kaggle/Linux actually does). On Windows,
`os.kill` with SIGTERM calls TerminateProcess and does NOT invoke a
registered Python signal handler, so it cannot be used to test this code
on a Windows dev machine. Tests instead call `handle_termination()`
directly -- that is exactly the function the registered POSIX handler
would call, so it exercises the identical code path deterministically and
portably.
"""
import json
import os
import signal
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch


class SigtermInterrupt(Exception):
    """Raised (not delivered as a raw OS signal) to unwind the training loop
    after a checkpoint has been safely written, so callers' try/finally
    blocks (e.g. scripts/train.py's budget-ledger logging) still run.
    """

    def __init__(self, step: int):
        super().__init__(f"SIGTERM received at step {step}; checkpoint saved.")
        self.step = step


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._installed = False
        self._orig_handler = None

    def _path_for_step(self, step: int) -> Path:
        return self.checkpoint_dir / f"ckpt_step{step:08d}.pt"

    def _latest_pointer(self) -> Path:
        return self.checkpoint_dir / "latest.json"

    def save(self, state: dict, step: int) -> Path:
        """Atomically write `state` as the checkpoint for `step` and update
        the `latest.json` pointer. Safe to call from a signal handler path.
        """
        target = self._path_for_step(step)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.checkpoint_dir), suffix=".pt.tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, target)  # atomic rename, same filesystem
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        fd2, tmp_name2 = tempfile.mkstemp(dir=str(self.checkpoint_dir), suffix=".json.tmp")
        os.close(fd2)
        tmp_path2 = Path(tmp_name2)
        try:
            tmp_path2.write_text(json.dumps({"step": step, "path": target.name}), encoding="utf-8")
            os.replace(tmp_path2, self._latest_pointer())
        except BaseException:
            tmp_path2.unlink(missing_ok=True)
            raise
        return target

    def load_latest(self) -> Tuple[Optional[dict], Optional[int]]:
        ptr = self._latest_pointer()
        if not ptr.exists():
            return None, None
        info = json.loads(ptr.read_text(encoding="utf-8"))
        ckpt_path = self.checkpoint_dir / info["path"]
        if not ckpt_path.exists():
            return None, None
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return state, info["step"]

    def handle_termination(self, get_state_fn: Callable[[], Tuple[dict, int]]) -> None:
        """Write a checkpoint from `get_state_fn() -> (state, step)` and
        raise SigtermInterrupt. This is the actual handler body; the
        registered signal.signal callback and tests both call this same
        function so behavior is identical regardless of trigger.
        """
        state, step = get_state_fn()
        self.save(state, step)
        raise SigtermInterrupt(step)

    def register_sigterm_handler(self, get_state_fn: Callable[[], Tuple[dict, int]]) -> None:
        """Install a SIGTERM handler on POSIX. On Windows this registers
        successfully (CPython allows it) but the OS will generally not
        deliver SIGTERM the way POSIX does -- see module docstring.
        """
        def _handler(signum, frame):
            self.handle_termination(get_state_fn)

        self._orig_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _handler)
        self._installed = True

    def restore_default_handler(self) -> None:
        if self._installed:
            signal.signal(signal.SIGTERM, self._orig_handler)
            self._installed = False
