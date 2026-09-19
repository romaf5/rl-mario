"""rl_games 1.6.5 patches applied by train.py before the runner is built.

Each one is driven by a custom key of params.config (popped by train.py, so
rl_games never sees it); on another rl_games version the source checks fail
and the patch is skipped with a warning instead of changing unknown code.

  * shuffle_minibatches (default True): rl_games builds the discrete agent's
    PPODataset with permute=False, so every minibatch was the same block of
    whole trajectories in every mini-epoch (32 envs x 128 steps at 4096, 8
    envs at the Mac's 1024). The agent already calls apply_permutation()
    each mini-epoch; this turns it on.
  * value_norm_clip (default 5.0 = rl_games): the value normaliser clamps
    normalised returns (targets) and the value head's output to +-5 sd. In
    the 4-2 run the warp's scaled return (47) sat at 6.4 sd above the mean
    (ceiling 37.5 at epoch 3500): the targets of the steps before the warp
    were cut and the critic could not represent the warp's value.
  * checkpoints load onto the CPU first (map_location): torch.load put a Mac
    checkpoint's tensors on MPS, and copying an MPS float32 buffer into the
    CPU float64 value normaliser was a silent no-op (mean 0 / var 1 kept,
    "All keys matched") when resuming a Mac checkpoint with --device cpu.
"""
import importlib.metadata
import inspect
from typing import Optional

import torch
from rl_games.algos_torch import running_mean_std as _rms
from rl_games.algos_torch import torch_ext

_applied = False
_VALUE_CLIP = 5.0
_RMS_PATCHED = False


def _rms_forward(self, input, denorm: bool = False, mask: Optional[torch.Tensor] = None):
    # rl_games 1.6.5 RunningMeanStd.forward, clamp bound from self.clip_sigma
    if self.training:
        if mask is not None:
            mean, var = torch_ext.get_mean_var_with_masks(input, mask)
        else:
            mean = input.mean(self.axis)
            var = input.var(self.axis, unbiased=False)
        self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
            self.running_mean, self.running_var, self.count, mean, var, input.size(0))
    if self.per_channel:
        if len(self.insize) == 3:
            current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
            current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
        elif len(self.insize) == 2:
            current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
            current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
        elif len(self.insize) == 1:
            current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
            current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var
    else:
        current_mean = self.running_mean
        current_var = self.running_var
    if denorm:
        y = torch.clamp(input, min=-self.clip_sigma, max=self.clip_sigma)
        y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
    else:
        if self.norm_only:
            y = input / torch.sqrt(current_var.float() + self.epsilon)
        else:
            y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
            y = torch.clamp(y, min=-self.clip_sigma, max=self.clip_sigma)
    return y


def _rms_source_ok():
    """The copied forward is only valid for the code it was copied from."""
    try:
        src = inspect.getsource(_rms.RunningMeanStd.forward)
    except (OSError, TypeError):
        return False
    return (src.count('torch.clamp(input, min=-5.0, max=5.0)') == 1
            and src.count('torch.clamp(y, min=-5.0, max=5.0)') == 1)


def post_init(agent, shuffle=True, value_clip=None):
    """Per-agent part: minibatch permutation and the value normaliser's clamp
    (set on the model's instance only, never on an observation normaliser)."""
    if shuffle and hasattr(agent, 'dataset') and not agent.dataset.is_rnn:
        agent.dataset.permute = True
    if value_clip is not None:
        m = getattr(agent.model, '_orig_mod', agent.model)
        vms = getattr(m, 'value_mean_std', None)
        if vms is not None and hasattr(vms, 'clip_sigma'):
            vms.clip_sigma = float(value_clip)


def apply(shuffle_minibatches=True, value_norm_clip=None):
    """Install the patches (idempotent). Returns the list of active ones."""
    global _applied, _VALUE_CLIP, _RMS_PATCHED
    version = importlib.metadata.version('rl_games')
    active = []
    if version != '1.6.5':
        print(f'[rl_games] version {version}: patches written for 1.6.5 skipped '
              f'(minibatch shuffling, value clip, CPU checkpoint loading)')
        return active
    if value_norm_clip is not None:
        _VALUE_CLIP = float(value_norm_clip)
    if not _applied:
        from rl_games.algos_torch import a2c_discrete
        if _rms_source_ok():                    # decided once, before the forward is replaced
            _RMS_PATCHED = True
            orig_init = _rms.RunningMeanStd.__init__

            def __init__(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                self.clip_sigma = 5.0            # float attribute: TorchScript-visible

            _rms.RunningMeanStd.__init__ = __init__
            _rms.RunningMeanStd.clip_sigma = 5.0     # instances created before the patch (eager)
            _rms.RunningMeanStd.forward = _rms_forward
        else:
            print('[rl_games] RunningMeanStd.forward differs from 1.6.5: value_norm_clip ignored')
        orig_agent_init = a2c_discrete.DiscreteA2CAgent.__init__

        def agent_init(self, *args, **kwargs):
            orig_agent_init(self, *args, **kwargs)
            post_init(self, apply.shuffle, _VALUE_CLIP if _RMS_PATCHED else None)

        a2c_discrete.DiscreteA2CAgent.__init__ = agent_init
        orig_safe_load = torch_ext.safe_load

        def safe_load(filename):
            # 1.6.5's safe_load, with the tensors mapped onto the CPU
            if hasattr(torch.serialization, 'safe_globals'):
                import numpy as np
                with torch.serialization.safe_globals({
                        'numpy.core.multiarray.scalar': np.core.multiarray.scalar,
                        'numpy.dtype': np.dtype,
                        'numpy.dtypes.Float32DType': lambda: np.dtype('float32')}):
                    return torch.load(filename, weights_only=False, map_location='cpu')
            return torch.load(filename, weights_only=False, map_location='cpu')

        safe_load.orig = orig_safe_load
        torch_ext.safe_load = safe_load
        _applied = True
    apply.shuffle = bool(shuffle_minibatches)
    active.append('minibatch shuffling' if shuffle_minibatches else 'minibatches unshuffled')
    active.append(f'value clip {_VALUE_CLIP:g} sd' if _RMS_PATCHED else 'value clip 5 sd (unpatched)')
    active.append('checkpoints load on the CPU')
    print('[rl_games] patches: ' + ', '.join(active))
    return active


apply.shuffle = True
