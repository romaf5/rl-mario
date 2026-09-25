"""Does the world model's W actually steer?

A small average error is not the same as a useful one. The value network taught us this
today: it scored 99.6% on its ranking test and cleared one level in four, because nearly
every label in its data was the same number. The world model's W error is 2.3 frames, which
sounds good, and the search inside it still wanders.

So ask the question the search asks. From one state, unroll the model along three lines --
the search's route, independent random actions, and one held input -- and read the W it
predicts for each. The route is a measuring stick here and nothing else: no gradient, no
hint, only a line we happen to know is good. If the model cannot tell it from noise, no
search inside the model can either.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.wmprobe --model smbzero/runs/wm4/wm.pt
"""
import argparse
import numpy as np
import torch
from ..common import MAX_DELAY, ROUTE, THREADS, Search, e2e_segments
from ..model import load as load_model

LINES = ('route', 'random', 'sticky', 'flip1', 'flip3')


def line(kind, opt, t, K, rng):
    if kind.startswith('flip'):
        # The route with one or three actions replaced: near-siblings, which is what the
        # search really compares. Telling a good line from noise is the easy question.
        acts = line('route', opt, t, K, rng)
        for i in rng.choice(K, int(kind[4:]), replace=False):
            acts[i] = rng.integers(0, 12)
        return acts
    if kind == 'route':
        acts = np.array(opt[t:t + K], np.uint8).copy()
        if len(acts) < K:
            acts = np.concatenate([acts, rng.integers(0, 12, K - len(acts)).astype(np.uint8)])
        return acts
    if kind == 'sticky':
        return np.full(K, rng.integers(0, 12), np.uint8)
    return rng.integers(0, 12, K).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--level', default='1-1')
    ap.add_argument('--states', type=int, default=256)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    s = Search(threads=THREADS)
    seg = {g['level']: g for g in e2e_segments(s)}[a.level]
    s.set_progress_route(ROUTE, seg['opt'], seg['start'])
    model, _ = load_model(a.model)
    model.eval()
    rng = np.random.default_rng(a.seed)
    K, opt = a.depth, seg['opt']

    pred = {k: [] for k in LINES}
    true = {k: [] for k in LINES}
    for _ in range(a.states):
        t = int(rng.integers(4, max(len(opt) - K - 1, 5)))
        _, before = s.replay(seg['start'], opt[:t - 4])
        obs, _, state = s.replay_obs(before, opt[t - 4:t])          # the real four-frame history
        stack = obs[-4:]
        for kind in LINES:
            acts = line(kind, opt, t, K, rng)
            x = torch.from_numpy(stack[None]).cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                _, _, _, _, ws = model.unroll(x, torch.from_numpy(acts.astype(np.int64))[None].cuda())
            pred[kind].append(float(ws[-1].float()[0]))
            p = s.progress_along(state, ROUTE, opt, acts, ref_start=seg['start'])
            true[kind].append(float(np.clip(p[min(K, len(p) - 1)] - p[0] + 4.0 * K, 0, 512)))

    print('[wmprobe] %s, %d states, %d steps ahead' % (a.level, a.states, K))
    print('  %-8s %9s %9s' % ('line', 'W said', 'W true'))
    for kind in LINES:
        print('  %-8s %9.1f %9.1f' % (kind, np.mean(pred[kind]), np.mean(true[kind])))
    for other in ('random', 'sticky', 'flip1', 'flip3'):
        pw = np.array(pred['route']) < np.array(pred[other])
        tw = np.array(true['route']) < np.array(true[other])
        both = tw.sum()
        print('  route beats %-7s: the model says so %4.1f%% of the time, it is true %4.1f%%, '
              'and where it is true the model agrees %4.1f%%'
              % (other, 100 * pw.mean(), 100 * tw.mean(),
                 100 * (pw & tw).sum() / max(both, 1)))


if __name__ == '__main__':
    main()
