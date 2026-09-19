#!/usr/bin/env python3
"""Differential audit of the native training env: runs the EXACT training env
of a config (optionally with a checkpoint's sampled policy and a saved archive,
both read-only), records every training step, and re-derives rewards, dones,
time-outs and episode labels from the raw game RAM with independent scalar
reference code. Every disagreement is listed; exit code 1 if any.

    python tools/audit_env.py --config configs/mario_ppo_native_42.yaml \\
        --checkpoint runs/<run>/nn/<ckpt>.pth --archive runs/<run>/archive.pkl --steps 2000

Without --checkpoint the policy is uniform random (checks the env alone);
without --archive the archive grows from empty. The env is built WITHOUT
archive_path, so nothing is ever written back. Run it before every launch
(3-5 minutes) and after every env change: a clean report is 0 violations.

Reference rules (the design in mario_rewards.py / mario_native_vecenv.py):
  * progress: per life and frame, pay min(x - highwater, cap) for new ground;
    teleport-held steps and death steps pay nothing
  * cells: +bonus once per (frame, x//x_bin, y//y_bin) cell per life
  * clear: 500 + per_extra * (levels - 1) on a confirmed on-route advance;
    every step of a wrong exit and the pending step of a clear pay 0
  * dones: death (per-life episodes), wrong exit, the paid exit out of the
    trained levels (end_on_stage_exit), unpaid cutoff, zombie, wrap
  * time_outs (value bootstrap): exactly the plain unpaid cutoffs
  * labels: an episode is a door episode iff it starts from the level's door
    state, a restart iff it starts from an archive state; with
    life_loss_reset every life is a fresh draw (no continuation lives)
  * the policy's actions reach the game unchanged; frame stacks shift by one
    frame per step and restart fresh at every episode boundary
"""
import argparse
import collections
import os
import pickle
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mario_native_vecenv import MarioNativeVecEnv  # noqa: E402

ADDR = [0x6D, 0x86, 0x75F, 0x75C, 0x760, 0x74E, 0x74F, 0x704, 0x3B8, 0x0E,
        0x1D, 0xB5, 0x75A, 0x770, 0x7F8, 0x7F9, 0x7FA, 0x71A, 0x71C, 0x09]
TERMS = ('progress', 'clear', 'cells', 'cell_bonus')
gp_of_name = lambda s: (int(s.split('-')[0]) - 1) * 4 + int(s.split('-')[1]) - 1


class _Recording(MarioNativeVecEnv):
    def _after_step(self):
        self.pre_ram = self.ram.copy()   # the RAM the step is scored on, before any reset
        return super()._after_step()


def load_policy(cfg, ckpt, device):
    import torch
    import device_support
    from rl_games.algos_torch import model_builder
    dev = device_support.resolve_device(device)
    cc = cfg['params']['config']
    model = model_builder.ModelBuilder().load(cfg['params']).build(
        {'actions_num': 12, 'input_shape': (84, 84, 4), 'num_seqs': 1, 'value_size': 1,
         'normalize_value': cc['normalize_value'], 'normalize_input': cc['normalize_input']})
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    vms = None
    if 'value_mean_std.running_mean' in sd:
        vms = (float(sd['value_mean_std.running_mean']), float(sd['value_mean_std.running_var']))
    return model.to(dev).eval(), dev, vms


def record(cfg, args):
    """Run the training env; return per-step arrays for the training envs."""
    import torch
    cc = cfg['params']['config']
    ec = dict(cc['env_config'])
    for k in ('archive_path', 'name', 'action_type'):
        ec.pop(k, None)
    ec['n_threads'] = min(int(ec.get('n_threads', 8)), 8)
    env = _Recording('audit', args.actors or cc['num_actors'], seed=args.seed, **ec)
    if args.archive:
        A = pickle.load(open(args.archive, 'rb'))
        env.archive = A
        env.cell_wins = {c: e[3] for c, e in A.items() if len(e) > 3}
        env.cell_tries = {c: e[4] for c, e in A.items() if len(e) > 4}
        env.explore_wins = {c: e[5] for c, e in A.items() if len(e) > 5 and e[5]}
        print(f'[audit] archive {len(A)} cells (read-only copy)', flush=True)
    obs = env.reset(); nt = env.n_train
    model = vms = None
    if args.checkpoint:
        model, dev, vms = load_policy(cfg, args.checkpoint, args.device)
    gen = torch.Generator().manual_seed(args.seed)
    R = collections.defaultdict(list)
    prev = obs.copy(); t0 = time.time()
    for s in range(args.steps):
        if model is not None:
            with torch.no_grad():
                res = model({'obs': torch.from_numpy(obs).to(dev), 'is_train': False})
            lg = res['logits'].float().cpu()
            R['val'].append(res['values'].float().cpu().numpy()[:, 0])
        else:
            lg = torch.zeros(nt, 12)
        a = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).squeeze(1).numpy()
        R['door0'].append(env.is_door[:nt].copy())
        obs, r, d, inf = env.step(a)
        R['pre'].append(env.pre_ram[:nt][:, ADDR].astype(np.int16))
        R['post'].append(env.ram[:nt][:, ADDR].astype(np.int16))
        R['act'].append(a.astype(np.int8)); R['env_act'].append(env.last_action[:nt].astype(np.int8))
        R['rew'].append(r.copy()); R['done'].append(d.copy())
        R['tout'].append(np.asarray(inf.time_outs, dtype=bool).copy())
        R['door1'].append(env.is_door[:nt].copy()); R['restart1'].append(env.was_restart[:nt].copy())
        sig = env.last_signals
        R['hold'].append(sig.hold[:nt].copy()); R['xeff'].append(sig.x[:nt].astype(np.int32))
        R['frame'].append(np.asarray(sig.frame)[:nt].copy()); R['page_reset'].append(sig.page_reset[:nt].copy())
        R['unpaid'].append(env.unpaid[:nt].copy())
        R['entered'].append(np.array([c is not None for c in env.entered_cell[:nt]]))
        for t in TERMS:
            R['t_' + t].append(np.asarray(env.last_terms.get(t, np.zeros(env.num_actors)))[:nt].astype(np.float32))
        R['stack_ok'].append((obs[..., :3] == prev[..., 1:]).all(axis=(1, 2, 3)))
        R['fresh_ok'].append((obs[..., :1] == obs).all(axis=(1, 2, 3)))
        prev = obs.copy()
        if s % 500 == 0:
            print(f'[audit] step {s}/{args.steps} {time.time() - t0:.0f}s archive {len(env.archive)}', flush=True)
    D = {k: np.array(v) for k, v in R.items()}
    # archive: every key must be the cell of its own states, and only trained levels are saved
    bad_keys, checked = [], 0
    train = set(ec.get('random_stages') or [])
    probe = MarioNativeVecEnv('audit_probe', 1, n_threads=1, **{
        k: v for k, v in ec.items() if k not in ('explorer_envs', 'self_restart_prob', 'n_threads')})
    probe.reset()
    keys = list(env.archive.keys())
    for c in keys[-args.check_cells:] if args.check_cells else keys:
        e = env.archive[c]
        for st in (e[0] if isinstance(e[0], list) else [e[0]]):
            probe.load_state(0, st)
            probe.lib.benv_obs(probe.env, 0, probe.obs_u8[0].ctypes.data, probe.ram[0].ctypes.data)
            checked += 1
            k2 = probe.cell_of(0)
            if k2 != c or (train and c[0] not in train):
                bad_keys.append((c, k2))
    probe.close(); env.close()
    return D, vms, (checked, bad_keys)


def check(cfg, D, vms, arch):
    cc = cfg['params']['config']; ec = cc['env_config']
    specs = {s['type']: s for s in (ec.get('reward') or [])}
    CAP = float(specs.get('first_visit_progress', {}).get('cap', 20))
    lc = specs.get('level_clear', {'base': ec.get('stage_bonus', 500), 'per_extra': 100})
    BASE, PER = float(lc.get('base', 500)), float(lc.get('per_extra', 100))
    cs = specs.get('first_visit_cells')
    CB, CXB, CYB = (float(cs.get('bonus', 2)), int(cs.get('x_bin', 64)), int(cs.get('y_bin', 32))) if cs else (0.0, 64, 32)
    UNPAID, GRACE = int(ec.get('unpaid_timeout', 250)), int(ec.get('page_reset_grace', 60))
    route = ec.get('route_levels') or ec.get('random_stages')
    ROUTE = {gp_of_name(s) for s in route} if (route and ec.get('full_game')) else None
    TRAIN = {gp_of_name(s) for s in ec.get('random_stages') or []}
    EOSE = bool(ec.get('end_on_stage_exit')) and bool(TRAIN)
    LLR = bool(ec.get('life_loss_reset', True))
    assert ec.get('full_game') and ec.get('episode_life', True), 'audit covers full_game per-life configs'

    A = {a: k for k, a in enumerate(ADDR)}
    PRE, POST = D['pre'].tolist(), D['post'].tolist()
    S, N = D['rew'].shape
    L = {k: D[k].tolist() for k in ('xeff', 'hold', 'frame', 'rew', 'done', 'tout', 'unpaid', 'door0', 'door1',
                                     'restart1', 'entered', 'page_reset', 'act', 'env_act', 'stack_ok', 'fresh_ok')}
    T = {t: D['t_' + t].tolist() for t in TERMS}
    door_env = MarioNativeVecEnv('door', 1, random_stages=list(ec.get('random_stages') or ['1-1']),
                                 full_game=True, n_threads=1)
    doors = set()
    for s in (ec.get('random_stages') or []):
        door_env.load_state(0, door_env.states[s])
        door_env.lib.benv_obs(door_env.env, 0, door_env.obs_u8[0].ctypes.data, door_env.ram[0].ctypes.data)
        doors.add(tuple(int(v) for v in door_env.ram[0, ADDR]))
    door_env.close()

    f = lambda r, a: r[A[a]]
    xof = lambda r: f(r, 0x6D) * 256 + f(r, 0x86)
    gpof = lambda r: min(max(f(r, 0x75F) * 4 + f(r, 0x75C), 0), 31)
    frameof = lambda r: (((gpof(r) * 65536 + f(r, 0x760) * 256 + f(r, 0x74F)) * 8 + f(r, 0x74E)) * 2 + f(r, 0x704))
    viol = collections.defaultdict(list)
    bad = lambda k, s, i, d='': viol[k].append((s, i, d))
    eps = []
    for i in range(N):
        st = None
        for s in range(S):
            done = L['done'][s][i]
            if st is not None:
                r = PRE[s][i]
                x, hold, fr = L['xeff'][s][i], L['hold'][s][i], frameof(r)
                life, gp, mode = f(r, 0x75A), gpof(r), f(r, 0x770)
                died = life == 0xFF or life < st['lives']
                exp = dict(progress=0.0, cells=0.0, clear=0.0)
                if not hold and not died:
                    hw = st['hw'].get(fr)
                    if hw is None:
                        st['hw'][fr] = x
                    elif x > hw:
                        exp['progress'] = min(x - hw, CAP); st['hw'][fr] = x
                    if CB:
                        key = (fr, x // CXB, f(r, 0x3B8) // CYB)
                        if key not in st['seen']:
                            st['seen'].add(key); exp['cells'] = CB
                inc = gp > st['prog']; confirm = inc and gp == st['pending']; delta = gp - st['prog']
                okc = confirm and delta <= 15; badw = f(r, 0x75F) > 7
                wrong = (okc and ROUTE is not None and gp not in ROUTE) or badw; good = okc and not wrong
                if okc:
                    st['prog'] = gp
                st['pending'] = gp if inc else -1
                if good:
                    exp['clear'] = BASE + PER * max(delta - 1, 0)
                for t in ('progress', 'cells', 'clear'):
                    if abs(T[t][s][i] - exp[t]) > 1e-3:
                        bad(f'{t}_term_mismatch', s, i, (round(T[t][s][i], 2), exp[t], 'death' if died else ''))
                leaving = (inc and not good) or wrong
                rew = L['rew'][s][i]
                tot = sum(T[t][s][i] for t in TERMS)
                if leaving and rew != 0:
                    bad('paid_on_leaving_step', s, i, rew)
                if not leaving and abs(rew - tot) > 1e-3:
                    bad('reward_not_sum_of_terms', s, i, (rew, tot))
                if died and rew > 0:
                    bad('paid_on_death_step', s, i, tuple(t for t in TERMS if T[t][s][i] > 0))
                if T['cell_bonus'][s][i] > 0 and not (L['door0'][s][i] or not ec.get('cell_bonus_door_only')):
                    bad('novelty_paid_to_non_door', s, i)
                if T['cell_bonus'][s][i] > 0 and not L['entered'][s][i]:
                    bad('novelty_without_cell_entry', s, i)
                paid = rew > 0; pr = L['page_reset'][s][i]
                st['unpaid'] = 0 if paid else st['unpaid'] + 1
                st['after_reset'] = (st['after_reset'] or pr) and not paid
                if pr and not paid:
                    st['unpaid'] = max(st['unpaid'], UNPAID - GRACE)
                # a level change in progress never times out (its pending step pays 0 by design)
                timeout = st['unpaid'] >= UNPAID and not inc and not wrong
                st['nongame'] = 0 if mode == 1 else st['nongame'] + 1
                zombie = st['nongame'] >= 8
                wrapped = gp < st['gp0'] or badw
                causes = dict(death=died, wrong_exit=wrong, stage_exit=EOSE and good and gp not in TRAIN,
                              timeout=timeout, zombie=zombie, wrapped=wrapped)
                if done != any(causes.values()):
                    bad('done_mismatch', s, i, (done, sorted(k for k, v in causes.items() if v)))
                exp_tout = timeout and not st['after_reset'] and not (died or zombie or wrapped)
                if L['tout'][s][i] != exp_tout:
                    bad('time_out_mismatch', s, i, (L['tout'][s][i], exp_tout, sorted(k for k, v in causes.items() if v)))
                if not done and L['unpaid'][s][i] != st['unpaid']:
                    bad('unpaid_counter_mismatch', s, i, (L['unpaid'][s][i], st['unpaid']))
                if L['frame'][s][i] != fr:
                    bad('signals_frame_mismatch', s, i, (L['frame'][s][i], fr))
                st['lives'] = life; st['len'] += 1; st['ret'] += rew
                st['cleared'] |= good
                if done:
                    st['end'] = 'clear' if st['cleared'] else next((k for k, v in causes.items() if v), '?')
                    eps.append(st)
            if L['env_act'][s][i] != L['act'][s][i]:
                bad('action_substituted', s, i)
            if not done and not L['stack_ok'][s][i]:
                bad('frame_stack_not_shifted', s, i)
            if done and not L['fresh_ok'][s][i]:
                bad('frame_stack_not_fresh', s, i)
            if done:
                r0 = POST[s][i]
                cont = POST[s][i] == PRE[s][i]           # no state was loaded: the next life continues
                ref = 'continuation' if cont else ('door' if tuple(r0) in doors else 'restart')
                if LLR and cont:
                    bad('life_continued_despite_life_loss_reset', s, i)
                exp_door, exp_restart = ref == 'door', ref == 'restart'
                if L['door1'][s][i] != exp_door or L['restart1'][s][i] != exp_restart:
                    bad('episode_label_wrong', s, i, (ref, 'door' if L['door1'][s][i] else '',
                                                      'restart' if L['restart1'][s][i] else ''))
                st = dict(label=ref, x0=xof(r0), gp0=gpof(r0), lives=f(r0, 0x75A), hw={frameof(r0): xof(r0)},
                          seen=set(), prog=gpof(r0), pending=-1, unpaid=0, after_reset=False, nongame=0,
                          len=0, ret=0.0, cleared=False, end=None)

    print(f'\n[audit] {S} steps x {N} training envs, {len(eps)} complete episodes checked')
    checked, bad_keys = arch
    for c, k2 in bad_keys:
        viol['archive_key_mismatch'].append((0, 0, (c, k2)))
    print(f'[audit] archive: {checked} states re-keyed, {len(bad_keys)} mismatches')
    print('\n=== invariant violations ===')
    if not viol:
        print('none')
    for k in sorted(viol, key=lambda k: -len(viol[k])):
        v = viol[k]
        print(f'{k:40s} {len(v):7d}   e.g. ' + '; '.join(f'step {a} env {b} {c}' for a, b, c in v[:3]))
    print('\n=== episodes ===')
    by = collections.defaultdict(list)
    for ep in eps:
        by[ep['label']].append(ep)
    steps = {k: sum(e['len'] for e in L_) for k, L_ in by.items()}; tot = max(sum(steps.values()), 1)
    for k, L_ in sorted(by.items()):
        ends = collections.Counter(e['end'] for e in L_)
        x0 = collections.Counter(e['x0'] // 100 * 100 for e in L_).most_common(3)
        print(f'{k:13s} n={len(L_):5d} steps {steps[k] / tot:6.1%} len {np.mean([e["len"] for e in L_]):6.1f} '
              f'return {np.mean([e["ret"] for e in L_]):8.1f} clear {np.mean([e["cleared"] for e in L_]):.3f} '
              f'ends {dict(ends)} start_x {x0}')
    if vms is not None and 'val' in D:
        gamma, scale = float(cc['gamma']), float(cc.get('reward_shaper', {}).get('scale_value', 1.0))
        lam = float(cc.get('tau', 0.95)); clip = float(cc.get('value_norm_clip', 5.0))
        mean, var = vms; sd = np.sqrt(var + 1e-5)
        val, rew = D['val'], D['rew'] * scale
        done, tout = D['done'].astype(np.float32), D['tout'].astype(np.float32)
        rp = rew + gamma * val * tout; ret = np.zeros_like(val); H = int(cc['horizon_length'])
        for c0 in range(0, S - 1, H):
            last = 0.0
            for t in range(min(c0 + H, S - 1) - 1, c0 - 1, -1):
                nn = 1.0 - done[t]
                last = rp[t] + gamma * val[t + 1] * nn - val[t] + gamma * lam * nn * last
                ret[t] = last + val[t]
        z = (ret[:S - 1] - mean) / sd
        cl = D['t_clear'][:S - 1] > 0
        print(f'\n[audit] value normaliser: ceiling {mean + clip * sd:.1f} scaled (clip {clip} sd); '
              f'targets above it {(z > clip).sum()} of {z.size}; clear steps {cl.sum()}, '
              f'their targets {np.round(ret[:S - 1][cl][:8], 1)}')
    return viol


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/mario_ppo_native_42.yaml')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--archive', default=None, help='saved archive pickle (read-only)')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--actors', type=int, default=None)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--check-cells', type=int, default=500, help='re-key the last N archive cells (0 = all)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    D, vms, arch = record(cfg, args)
    viol = check(cfg, D, vms, arch)
    sys.exit(1 if viol else 0)


if __name__ == '__main__':
    main()
