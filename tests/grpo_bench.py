"""Checks for the GRPO trainer's update math and bookkeeping (no training).
Run: venv_retro/bin/python tests/grpo_bench.py

Properties (2026-09-11 review):
  * the stored rollout log-prob is the true log-prob (an asymmetric clamp
    against the fresh log-prob turned the ratio of every rare forced /
    hinted action into ~0, so no gradient could lift it);
  * the PPO ratio is bounded by clamping the DIFFERENCE, so lp == olp gives
    ratio 1 however small both are;
  * the self-imitation loss keeps a gradient below p = e^-20;
  * zero-advantage samples (dead groups) are not trained on (they only
    added entropy and flattened mastered levels);
  * --no-std applies in the reward-to-go path as well;
  * CLI reward knobs default to the config's env_config, and the launch is
    recorded in the run dir;
  * loop metrics read the flag that is actually set at the done step;
  * prompts pruned from the archive are pruned from the prompt pool;
  * a train-mode forward for logits does not update the value normaliser.
"""
import json, os, sys, tempfile
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grpo import train_grpo as G

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)))


# ---------------------------------------------------------------- log-probs / ratio / BC
logits = torch.zeros(1, 12); logits[0, 0] = 30.0            # action 0 near-certain; the rest ~e^-30
dist = torch.distributions.Categorical(logits=logits)
act = torch.tensor([5])
true_lp = dist.log_prob(act)
check('rollout log-prob is the true log-prob (%.1f), not a clamp' % true_lp.item(),
      torch.allclose(G.rollout_logp(dist, act), true_lp), G.rollout_logp(dist, act).item())
lp = torch.tensor([-25.0], requires_grad=True); olp = torch.tensor([-25.0])
ratio = torch.exp(G.log_ratio(lp, olp))
check('ratio is 1 when the fresh and stored log-probs agree at -25', torch.allclose(ratio, torch.ones(1)), ratio.item())
(ratio * 1.0).sum().backward()
check('policy gradient at a rare action is nominal (d ratio / d lp = 1)', abs(lp.grad.item() - 1.0) < 1e-5, lp.grad.item())
check('log-ratio is bounded', G.log_ratio(torch.tensor([0.0]), torch.tensor([-100.0])).item() <= 10.0 + 1e-6)
lps = torch.tensor([-30.0, -30.0], requires_grad=True)
G.bc_loss(lps).backward()
check('self-imitation loss keeps a gradient at log-prob -30', float(lps.grad.abs().sum()) > 0, lps.grad)

# ---------------------------------------------------------------- valid samples
mask = torch.tensor([1.0, 1.0, 0.0, 1.0]); adv = torch.tensor([0.5, 0.0, 0.7, -0.2])
vi = G.valid_indices(mask, adv).tolist()
check('valid samples: masked-out and zero-advantage samples are excluded', vi == [0, 3], vi)

# ---------------------------------------------------------------- advantages
R = np.array([1.0, 3.0, 5.0, 7.0, 2.0, 2.0, 2.0, 2.0], np.float32)     # 2 groups of 4; group 2 has zero variance
adv, live = G.outcome_advantages(R, groups=2, group=4, no_std=False)
check('outcome advantages: zero-variance group -> 0, live groups counted', live == 1 and np.all(adv[4:] == 0) and abs(adv[:4].mean()) < 1e-6, (live, adv))
adv_ns, _ = G.outcome_advantages(R, groups=2, group=4, no_std=True)
check('outcome advantages: --no-std leaves the centred returns undivided', np.allclose(adv_ns[:4], R[:4] - 4.0), adv_ns)
Gt = np.stack([R, R * 2.0])                                          # H=2
a_std = G.rtg_advantages(Gt, groups=2, group=4, no_std=False)
a_ns = G.rtg_advantages(Gt, groups=2, group=4, no_std=True)
check('rtg advantages: --no-std applies (centred, undivided) and dead groups are 0',
      np.allclose(a_ns[0, :4], R[:4] - 4.0) and np.allclose(a_ns[1, :4], 2 * R[:4] - 8.0) and np.all(a_ns[:, 4:] == 0)
      and not np.allclose(a_std[0, :4], a_ns[0, :4]), (a_std[0], a_ns[0]))

# ---------------------------------------------------------------- CLI defaults follow the config
cfg_ec = {'cell_bonus': 100, 'cell_x_bin': 64}
ov = G.resolve_env_overrides(cell_bonus=None, cell_x_bin=None, env_config=cfg_ec)
check('CLI reward knobs default to the config (bonus 100, x-bin 64)', ov == {'cell_bonus': 100.0, 'cell_x_bin': 64}, ov)
ov = G.resolve_env_overrides(cell_bonus=0.0, cell_x_bin=None, env_config=cfg_ec)
check('an explicit CLI value still overrides the config', ov == {'cell_bonus': 0.0, 'cell_x_bin': 64}, ov)
ov = G.resolve_env_overrides(cell_bonus=None, cell_x_bin=None, env_config={})
check('without config values the old defaults apply (0, 128)', ov == {'cell_bonus': 0.0, 'cell_x_bin': 128}, ov)
d = tempfile.mkdtemp()
G.write_launch_record(d, ['train_grpo.py', '--rtg'], {'cell_bonus': 100.0}, {'rtg': True})
rec = json.load(open(os.path.join(d, 'launch.json')))
check('launch record holds argv, resolved env config and args', rec['argv'][1] == '--rtg' and rec['env_config']['cell_bonus'] == 100.0 and rec['args']['rtg'] is True, rec)

# ---------------------------------------------------------------- loop flag
check('looped(): reads page_resets / loop_timeout, not the drop-step flag',
      G.looped({'page_resets': 1, 'page_reset': False}) and G.looped({'loop_timeout': True}) and not G.looped({'page_reset': False, 'page_resets': 0}))

# ---------------------------------------------------------------- prompt pruning
class _Env:
    stages = ['8-4']; states = {'8-4': b'door'}; cell_x_bin = 64
p = G.Prompts(_Env(), None, 0.25)
K = lambda b: ('8-4', 3, b, 2, 0, 3, 0)
p.refresh({K(10): [[b'a'], 0, 300], K(11): [[b'b'], 0, 300], K(12): [[b'c'], 0, 300]})
p.score[:] = [1.0, 2.0, 3.0]; p.uses[:] = [4, 5, 6]
p.refresh({K(10): [[b'a'], 0, 300], K(12): [[b'c2'], 0, 300]})
check('prompts: a cell evicted from the archive leaves the prompt pool (scores/uses stay aligned)',
      p.cells == [K(10), K(12)] and list(p.score) == [1.0, 3.0] and list(p.uses) == [4, 6] and p.states[1] == [b'c2'],
      (p.cells, list(p.score), list(p.uses)))

# ---------------------------------------------------------------- value normaliser untouched by logits
params = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'mario_ppo_native_84.yaml')))['params']
model = G.build_model(params, params['config'], (84, 84, 4))
model.train()
before = model.value_mean_std.count.clone()
G.logits_of(model, torch.zeros(2, 84, 84, 4))
check('logits_of in train mode leaves the value normaliser count unchanged', torch.equal(model.value_mean_std.count, before),
      (before.item(), model.value_mean_std.count.item()))

# ---------------------------------------------------------------- the trainer's evals sample the policy, seeded
from mario_native_vecenv import MarioNativeVecEnv
rp = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'mario_ppo_native_routeDet.yaml')))['params']
rc = rp['config']; ec = dict(rc['env_config']); [ec.pop(k, None) for k in ('name', 'action_type', 'archive_path', 'video_levels')]
ec.update(sticky_actions=0.0, explore_eps=0.0, self_restart_prob=0.0, explore_episode_prob=0.0, reset_noops=0, n_threads=2,
          dense_infos=True, seed=1, random_stages=['1-1'], route_levels=list(rc['env_config']['random_stages']))
eval_env = MarioNativeVecEnv('t', 4, **ec); eval_env.reset()
model.eval(); dev = torch.device('cpu')
ev1 = G.clean_door_eval(model, eval_env, dev, 4, max_steps=30, seed=5)
ev2 = G.clean_door_eval(model, eval_env, dev, 4, max_steps=30, seed=5)
ev3 = G.clean_door_eval(model, eval_env, dev, 4, max_steps=30, seed=6)
eval_env.close()
check('door eval samples the policy (per-episode max x are not all equal)', len(set(ev1['max_x_all'])) > 1, ev1.get('max_x_all'))
check('door eval is reproducible for a fixed seed', ev1 == ev2, (ev1, ev2))
check('door eval differs for another seed', ev1 != ev3, (ev1, ev3))
fg1 = G.full_game_eval(model, rc, dev, 2, max_steps=30, n_threads=2, seed=3)
fg2 = G.full_game_eval(model, rc, dev, 2, max_steps=30, n_threads=2, seed=3)
check('full-game eval is reproducible for a fixed seed', fg1 == fg2, (fg1, fg2))

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
