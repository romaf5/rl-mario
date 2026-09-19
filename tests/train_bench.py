"""PPO-side checks for rlg_patches.py (rl_games 1.6.5) and train.py (2026-09-19).
Run: venv_retro/bin/python tests/train_bench.py

Properties:
  * minibatch shuffling: the agent's PPODataset permutes every mini-epoch,
    and rl_games' permutation keeps every tensor of a sample aligned;
  * the value normaliser's clamp is configurable on the MODEL's value
    normaliser (TorchScript-compiled), and without the key its outputs equal
    rl_games' own (+-5 sd);
  * checkpoints load onto the CPU (a Mac checkpoint's MPS tensors copied into
    the CPU float64 normaliser was a silent no-op);
  * train.py pops the custom keys before rl_games sees them.
"""
import os, sys, tempfile, types
import torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from rl_games.algos_torch import model_builder, running_mean_std as rms, torch_ext
from rl_games.common import datasets

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'mario_ppo_native_42.yaml')))
params, cc = cfg['params'], cfg['params']['config']
orig_rms = rms.RunningMeanStd((1,))            # rl_games' own, before any patch
orig_forward = rms.RunningMeanStd.forward


def build():
    m = model_builder.ModelBuilder().load(params).build(
        {'actions_num': 12, 'input_shape': (84, 84, 4), 'num_seqs': 1, 'value_size': 1,
         'normalize_value': True, 'normalize_input': False})
    m.eval()
    m.value_mean_std.running_mean.fill_(2.0); m.value_mean_std.running_var.fill_(9.0)
    return m


import rlg_patches
active = rlg_patches.apply(shuffle_minibatches=True, value_norm_clip=10.0)
check('patches: all three active on rl_games 1.6.5 (%s)' % active,
      any('shuffling' in a for a in active) and any('value clip 10' in a for a in active))

# ---------------------------------------------------------------- value normaliser clamp
m = build()
x = torch.tensor([[7.0], [-7.0], [1.5]])
default = m.value_mean_std(x, denorm=True)
check('value clip: without the key the output clamps at +-5 sd exactly like rl_games (%s)' % default.flatten().tolist(),
      torch.allclose(default.flatten(), torch.tensor([2 + 3 * 5.0, 2 - 3 * 5.0, 2 + 3 * 1.5])))
agent = types.SimpleNamespace(model=m, dataset=datasets.PPODataset(64, 16, True, False, 'cpu', 1))
rlg_patches.post_init(agent, shuffle=True, value_clip=10.0)
out = m.value_mean_std(x, denorm=True)
check('value clip: the model\'s value normaliser clamps at +-10 sd after post_init (%s)' % out.flatten().tolist(),
      torch.allclose(out.flatten(), torch.tensor([2 + 3 * 7.0, 2 - 3 * 7.0, 2 + 3 * 1.5])))
ret = torch.tensor([[2 + 3 * 7.0]])
check('value clip: a return 7 sd above the mean is normalised to 7, not cut to 5 (%.2f)' % m.value_mean_std(ret).item(),
      abs(m.value_mean_std(ret).item() - 7.0) < 1e-4)
orig_rms.running_mean.fill_(2.0); orig_rms.running_var.fill_(9.0); orig_rms.eval()
m2 = build()
same = all(torch.allclose(m2.value_mean_std(v, denorm=d), orig_forward(orig_rms, v, denorm=d))
           for v in (torch.linspace(-9, 9, 37)[:, None],) for d in (False, True))
check('value clip: the patched forward equals rl_games\' own at the default clip (norm and denorm)', same)

# ---------------------------------------------------------------- minibatch shuffling
check('shuffle: post_init turns the dataset permutation on', agent.dataset.permute is True)
ds = datasets.PPODataset(64, 16, True, False, 'cpu', 1, permute=True)
base = torch.arange(64)
ds.update_values_dict({'obs': base.float()[:, None].repeat(1, 3), 'actions': base * 2, 'returns': base.float() + 0.5,
                       'rnn_states': None, 'mu': None})
ds.apply_permutation()
v = ds.values_dict
aligned = torch.equal(v['actions'], v['obs'][:, 0].long() * 2) and torch.equal(v['returns'], v['obs'][:, 0] + 0.5)
check('shuffle: rl_games\' permutation keeps every sample aligned across tensors', aligned)
check('shuffle: the order changes', not torch.equal(v['actions'], base * 2))
first = ds[0]['actions'].clone(); ds.apply_permutation()
check('shuffle: each mini-epoch draws other minibatches', not torch.equal(first, ds[0]['actions']))

# ---------------------------------------------------------------- checkpoints load on the CPU
dev = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else None)
if dev:
    path = os.path.join(tempfile.mkdtemp(), 'ck.pth')
    torch.save({'model': {'value_mean_std.running_mean': torch.tensor([4.0], device=dev)}}, path)
    ck = torch_ext.load_checkpoint(path)
    check('checkpoint: a %s checkpoint loads onto the CPU' % dev, ck['model']['value_mean_std.running_mean'].device.type == 'cpu')
else:
    print('SKIP checkpoint: no MPS / CUDA device')

# ---------------------------------------------------------------- train.py pops its keys
import train
src = open(os.path.join(ROOT, 'train.py')).read()
check('train.py: shuffle_minibatches / value_norm_clip are popped before rl_games sees them',
      "pop('shuffle_minibatches'" in src and "pop('value_norm_clip'" in src)

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
