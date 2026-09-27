"""4-2: from route states before the hidden block, does the search bump it (the vine spawns) within 80 decisions?

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.vinetest relv3 relv_mc235
"""
import sys, numpy as np
from smbzero.common import ROUTE, THREADS, Search, e2e_segments
from smbzero.net import RelEvaluator, load
from smbzero.play import Game, Player
from smbzero.relvalue import load as load_rel
s = Search(threads=THREADS); seg = {g['level']: g for g in e2e_segments(s)}['4-2']
opt = np.asarray(seg['opt'], np.uint8)
net = load('smbzero/runs/zero8/net.pt')[0]
N = 80
for name in sys.argv[1:]:
    ev = RelEvaluator(net, load_rel('smbzero/runs/%s/relvalue.pt' % name)[0], max_leaves=128 * 8)
    player = Player(s, ev, 8, per_tree=128, value_mix=1.0, min_backup=True, relative=True)
    for t0 in (60, 90, 110, 120):
        _, st = s.replay(seg['start'], opt[:t0])
        games = [Game(s.frames(st, d), tag=d) for d in range(8)]
        player.play(games, sims=1000, segment_limit=1, max_decisions=N)
        res = []
        for g in games:
            st2, hit = g.start, None
            for i, a in enumerate(g.actions):
                tr, st2 = s.replay(st2, np.array([a], np.uint8))
                if 0x2F in s.ram(st2)[0x16:0x1B]: hit = i; break
            res.append('-' if hit is None else str(hit))
        print('%-8s from route t=%3d (route bumps at +%d): vine at %s' % (name, t0, 131 - t0, ' '.join(res)), flush=True)
