"""Replay every demo of a prompts.pkl and drop the ones that no longer enter
their target (explorer walks that continued past a death used to be
recorded as demos). Writes <out>."""
import sys, pickle, argparse, yaml, numpy as np; sys.path.insert(0, '.')
from mario_native_vecenv import MarioNativeVecEnv
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--config', default='configs/mario_ppo_native_84.yaml'); a = ap.parse_args()
ec = dict(yaml.safe_load(open(a.config))['params']['config']['env_config']); ec.pop('name', None); ec.pop('action_type', None)
ec.update(dict(self_restart_prob=1e-6, explore_eps=0.0, explore_episode_prob=0.0, archive_path=None, cell_tiles=True, cell_y_band=32, cell_max_variants=3, sticky_actions=0.0, n_threads=1, dense_infos=True, reset_noops=0))
env = MarioNativeVecEnv('clean', 1, **ec); env.reset()
z = pickle.load(open(a.src, 'rb'))
def ok(cell, acts, start):
    env.lib.benv_load(env.env, 0, start); env._fetch_obs(0); env._post_reset_init([0], env.ram); env.ep_cells[0] = set(); env.start_cell[0] = None
    for act in acts:
        env.step(np.array([int(act)]))
        if env.entered_cell[0] == cell:
            return True
    return False
for key in ('demos', 'grad'):
    keep = {}
    for c, v in z[key].items():
        ents = [e for e in (v if isinstance(v, list) else [v]) if ok(c, e[1], e[2])]
        if ents:
            keep[c] = ents
    print('%s: kept %d of %d' % (key, len(keep), len(z[key]))); z[key] = keep
z['graduated'] = len(z['grad']); env.close()
pickle.dump(z, open(a.out, 'wb')); print('wrote', a.out)
