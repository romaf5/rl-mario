"""Torch device selection shared by the trainers (train.py, grpo/train_grpo.py).

Configs say `device: cuda:0` (the GPU box). On a machine without CUDA -- an
Apple Silicon Mac -- the trainers fall back to Apple's MPS backend (or the
CPU) instead of crashing. On MPS, rl_games' value normaliser keeps its
running statistics in float32: its float64 buffers cannot live on MPS. On a
CUDA machine nothing changes.
"""
import torch

_mps_patched = False


def resolve_device(requested=None):
    """Return the device string to train on for `requested` (config value or
    CLI override). A CUDA device without CUDA falls back to mps, then cpu."""
    dev = (requested or 'cuda:0').strip()
    if dev == 'auto':
        dev = 'cuda:0' if torch.cuda.is_available() else 'cuda'
    if dev.startswith('cuda') and not torch.cuda.is_available():
        fallback = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'[device] CUDA is not available: training on {fallback} '
              f'instead of {dev}')
        dev = fallback
    if dev.startswith('mps'):
        apply_mps_patches()
    return dev


def apply_mps_patches():
    """rl_games' RunningMeanStd registers float64 buffers; MPS has no float64.
    float32 statistics are ample for the value normaliser (per-update relative
    increments stay far above float32 precision)."""
    global _mps_patched
    if _mps_patched:
        return
    from rl_games.algos_torch import running_mean_std as rms
    orig_init = rms.RunningMeanStd.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for name in ('running_mean', 'running_var', 'count'):
            setattr(self, name, getattr(self, name).float())

    rms.RunningMeanStd.__init__ = __init__
    _mps_patched = True
