"""Checks for the observer's evaluation path (no training, CPU, random model).
Run: venv_retro/bin/python tests/eval_bench.py

Properties (2026-09-11 review):
  * the video env is the native core stepping hack-free: the policy, the
    timer hack and the rendered frames all live in one emulator (the
    lockstep renderer diverged after the first forced time-up);
  * a clip's trace records the stepping mode it was recorded with;
  * per-level evaluation plays the SAMPLED policy from each level's door,
    N episodes with a fixed seed: a rate per level, reproducible, and the
    episodes are not clones of each other; each level is evaluated on its
    own door (the old door eval drew a random level per env);
  * the sequential full-game evaluation is sampled and seeded too.
"""
import os, sys, tempfile, types
import numpy as np, torch, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from rl_games.algos_torch import model_builder
from callbacks import MarioObserver
from mario_native_vecenv import NativeEvalEnv

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


class FakeWriter:
    def __init__(self, logdir):
        self.logdir = logdir; self.scalars = {}
    def add_scalar(self, tag, v, step):
        self.scalars.setdefault(tag, []).append((step, float(v)))
    def flush(self):
        pass


params = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'mario_ppo_native_routeDet.yaml')))['params']
cfg = params['config']
net = model_builder.ModelBuilder().load(params)
torch.manual_seed(0)
model = net.build({'actions_num': 12, 'input_shape': (84, 84, 4), 'num_seqs': 1, 'value_size': 1,
                   'normalize_value': cfg['normalize_value'], 'normalize_input': cfg['normalize_input']}).eval()
run_dir = tempfile.mkdtemp()
obs = MarioObserver(video_freq=0, eval_env_kwargs=dict(cfg['eval_env_config']))
obs.algo = types.SimpleNamespace(env_config=dict(cfg['env_config']), is_rnn=False)
obs.writer = FakeWriter(os.path.join(run_dir, 'summaries'))

# ---------------------------------------------------------------- video env
env = obs._make_eval_env(random_stages=['1-1'], full_game=True)
check('video env is the native single-env adapter', isinstance(env, NativeEvalEnv), type(env).__name__)
check('video env steps hack-free', getattr(env.v, '_raw_steps', False) is True)
env.close()
ek = dict(cfg['eval_env_config']); ek['backend'] = 'lockstep'
obs2 = MarioObserver(video_freq=0, eval_env_kwargs=ek); obs2.algo = obs.algo; obs2.writer = obs.writer
env = obs2._make_eval_env(random_stages=['1-1'], full_game=True)
check('a lockstep backend request is mapped to the native adapter', isinstance(env, NativeEvalEnv), type(env).__name__)
env.close()

# ---------------------------------------------------------------- trace stamps its stepping mode
for ep, raw in ((7, True), (8, False)):
    env = obs._make_eval_env(random_stages=['1-1'], full_game=True, raw_steps=raw)
    fr, pf, st, info, tot, ps = obs._play_clip(model, env, 12, ep, stop_on_level_change=True)
    tr = [f for f in os.listdir(os.path.join(run_dir, 'eval_traces', f'epoch_{ep}')) if f.endswith('.npz')]
    z = np.load(os.path.join(run_dir, 'eval_traces', f'epoch_{ep}', tr[0]))
    check('clip trace is stamped with its stepping mode (raw=%d)' % int(raw), len(tr) == 1 and int(z['raw']) == int(raw), (tr, int(z['raw']) if tr else None))
    check('clip info is the env info dict (max_x_pos=%s)' % info.get('max_x_pos'), info.get('max_x_pos', 0) >= 40 and 'life' in info, info)

# ---------------------------------------------------------------- clips play the SAMPLED policy, seeded
def clip_actions(ep, seed):
    env = obs._make_eval_env(random_stages=['1-1'], full_game=True)
    obs._play_clip(model, env, 40, ep, stop_on_level_change=True, seed=seed)
    tr = [f for f in os.listdir(os.path.join(run_dir, 'eval_traces', f'epoch_{ep}')) if f.endswith('.npz')]
    return list(np.load(os.path.join(run_dir, 'eval_traces', f'epoch_{ep}', tr[0]))['actions'])
a1, a2, a3 = clip_actions(21, 5), clip_actions(22, 5), clip_actions(23, 6)
check('clip: same seed -> identical action sequence (reproducible)', a1 == a2, (a1[:10], a2[:10]))
check('clip: the policy is sampled (another seed -> different actions; argmax of a random net would repeat one action)',
      a1 != a3 and len(set(a1)) > 1, (a1[:10], a3[:10]))

# ---------------------------------------------------------------- per-level sampled eval
res = obs._level_eval(model, 9, levels=['1-1', '8-1'], n=4, max_steps=40, seed=3)
sc = obs.writer.scalars
check('level eval writes a clear rate per level', 'eval/level_clear/1-1' in sc and 'eval/level_clear/8-1' in sc, sorted(sc)[:8])
check('level eval writes max_x mean / timeout / wrong-exit / death rates per level',
      all(f'eval/level_{k}/1-1' in sc for k in ('max_x_mean', 'timeout_rate', 'wrong_exit_rate', 'death_rate')), sorted(sc)[:12])
res2 = obs._level_eval(model, 10, levels=['1-1', '8-1'], n=4, max_steps=40, seed=3)
check('level eval is reproducible for a fixed seed', res == res2, (res, res2))
res3 = obs._level_eval(model, 11, levels=['1-1'], n=4, max_steps=40, seed=4)
check('level eval samples the policy (a different seed gives different episodes)', res['1-1'] != res3['1-1'], (res['1-1'], res3['1-1']))
check('level eval episodes are not clones (per-episode max x differ)', len(set(res['1-1']['max_x'])) > 1, res['1-1']['max_x'])
idx = os.path.join(run_dir, 'eval_traces', 'epoch_9', 'index.csv')
rows = open(idx).read().splitlines() if os.path.exists(idx) else []
check('level eval index lists the level of every episode', len(rows) > 1 and rows[0].startswith('level,') and any(r.startswith('8-1,') for r in rows[1:]), rows[:3])

# ---------------------------------------------------------------- a cleared level counts even if the episode plays on
import csv
ACT = ['NOOP', 'R', 'R+A', 'R+B', 'R+A+B', 'A', 'L', 'L+A', 'L+B', 'L+A+B', 'DOWN', 'UP']; AIDX = {a: i for i, a in enumerate(ACT)}
class TraceModel:
    """Forces a recorded action sequence through the eval's sampler (one-hot logits)."""
    def __init__(self, acts): self.acts, self.t = acts, 0
    def __call__(self, d):
        lg = torch.full((d['obs'].shape[0], 12), -1e9); lg[:, self.acts[min(self.t, len(self.acts) - 1)]] = 0.0; self.t += 1
        return {'logits': lg}
rows = list(csv.DictReader(open(os.path.join(ROOT, 'traces', 'play_0905-232759.csv'))))     # human: 1-1 -> 1-2 -> 4-1
acts = [AIDX[r['action']] for r in rows]
first_12 = next(i for i, r in enumerate(rows) if r['level'] == '1-2')
res = obs._level_eval(TraceModel(acts), 14, levels=['1-1'], n=1, max_steps=first_12 + 60, seed=0)
check('level eval: a level cleared mid-episode counts as cleared although the episode plays on into the next level (trace clears 1-1 at step %d)' % first_12,
      res['1-1']['clear'] == 1.0 and res['1-1']['ends']['clear'] == 1.0, res['1-1'])

# ---------------------------------------------------------------- route-aware progress
route_gps = {0, 1, 12, 13, 28, 29, 30, 31}
check('route progress: an off-route exit (4-2 flag -> 4-3) counts as the last on-route level (4-2)',
      MarioObserver.route_progress(14, route_gps) == 13 and MarioObserver.route_progress(2, route_gps) == 1
      and MarioObserver.route_progress(28, route_gps) == 28 and MarioObserver.route_progress(0, route_gps) == 0,
      [MarioObserver.route_progress(g, route_gps) for g in (14, 2, 28, 0)])

# ---------------------------------------------------------------- sequential sampled eval
seq = obs._sequential_eval(model, 12, n=2, max_steps=30, seed=1)
check('sequential eval writes sampled game progress mean/max, victory rate and off-route exit rate',
      all(t in sc for t in ('eval/game_progress_sampled_mean', 'eval/game_progress_sampled_max', 'eval/victory_rate_sampled', 'eval/off_route_exit_rate_sampled')), sorted(t for t in sc if 'sampled' in t))
seq2 = obs._sequential_eval(model, 13, n=2, max_steps=30, seed=1)
check('sequential eval is reproducible for a fixed seed', seq == seq2, (seq, seq2))

# ---------------------------------------------------------------- single-level runs evaluate what they train
obs42 = MarioObserver(video_freq=0, eval_env_kwargs=dict(cfg['eval_env_config']))
ec42 = dict(cfg['env_config']); ec42['random_stages'] = ['4-2']; ec42['route_levels'] = list(cfg['env_config']['random_stages'])
obs42.algo = types.SimpleNamespace(env_config=ec42, is_rnn=False); obs42.writer = FakeWriter(os.path.join(run_dir, 'summaries42'))
r42 = obs42._level_eval(model, 30, n=2, max_steps=20, seed=1)
check('level eval defaults to the TRAINED levels (a 4-2 run evaluates 4-2 only, not the whole route)', list(r42) == ['4-2'], list(r42))
s42 = obs42._sequential_eval(model, 31, n=2, max_steps=20, seed=1)
check('sequential eval starts from the first trained level (a 4-2 run: progress index 13 at the start)', s42['progress_max'] == 13 and s42['progress_mean'] == 13.0, s42)

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
