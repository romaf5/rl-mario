"""Shared paths and constants for SMBZero. Hardware: GPU 1 only, 32 CPU threads."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'search', 'python'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')          # the one GPU SMBZero may use

from smbsearch import Search, Forest, load_state, ROUTE, FPS, FRAME_SKIP, gp, level_name  # noqa: E402

THREADS = 32
V_SCALE = 4096.0            # frames: the net's value range; a dead end is worth V_SCALE
DATA = os.path.join(HERE, 'data')
RUNS = os.path.join(HERE, 'runs')
E2E = os.path.join(REPO, 'search', 'out', 'e2e', 'route.npz')   # search/full_game.sh
MAX_DELAY = 60              # the start delay at 1-1: 0..60 NOOP frames


def e2e_segments(s):
    """The verified full-game search route per level: entry state (first in-control
    frame), its explore reference and its optimised actions."""
    z = np.load(E2E)
    state = load_state(str(z['start']))
    _, state = s.replay(state, np.zeros(int(z['lead_in']), np.uint8))
    segs, i = [], int(z['lead_in'])
    acts = z['actions']
    for k, lvl in enumerate(z['levels']):
        n, settle = int(z['seg_actions'][k]), int(z['seg_settle'][k])
        segs.append(dict(level=str(lvl), start=state, ref=z['ref_actions_%d' % k], opt=acts[i:i + n]))
        _, state = s.replay(state, acts[i:i + n + settle])
        i += n + settle
    return segs


def episode(frames, policy, value, forced, **meta):
    """One segment of play as training data: frames (n+1, 84, 84) with frames[0] the
    start screen and frames[t+1] the frame after decision t; per decision t the policy
    target (12,), value target (frames to go) and the forced flag."""
    return dict(frames=np.asarray(frames, np.uint8), policy=np.asarray(policy, np.float32),
                value=np.asarray(value, np.float32), forced=np.asarray(forced, bool), **meta)


def save_episode(path, ep):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **ep)


def route_values(segs):
    """{level: (entry state, optimised actions)}: the routes MCTS leaf values follow."""
    return {l: (g['start'], g['opt']) for l, g in segs.items()}
