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

    # macOS aborts a GPU command buffer that runs too long while the display
    # needs the GPU (kIOGPUCommandBufferCallbackErrorImpactingInteractivity,
    # then "victim of GPU error/recovery" for queued work), and PyTorch goes on
    # with whatever the aborted ops left behind. Sync after every minibatch
    # update (no change to the maths). NB with use_diagnostics (every config)
    # rl_games already reads exp_var / clip_frac to the CPU after each update,
    # which drains the queue too, so this only matters with diagnostics off;
    # what shortened each buffer was --minibatch-size 1024 (~350 -> ~90 ms).
    from rl_games.algos_torch import a2c_discrete, a2c_continuous
    for cls in (a2c_discrete.DiscreteA2CAgent, a2c_continuous.A2CAgent):
        orig_train = cls.train_actor_critic

        def train_actor_critic(self, *args, _orig=orig_train, **kwargs):
            out = _orig(self, *args, **kwargs)
            torch.mps.synchronize()
            return out

        cls.train_actor_critic = train_actor_critic
    _mps_patched = True
