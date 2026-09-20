"""MCTS forest checks with a stand-in net (uniform priors, constant value).

  venv_retro/bin/python search/tests/mcts_test.py [threads]
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SEARCH, 'python'))
from smbsearch import Search, Forest, load_state, ROUTE


def think(f, trees, sims, per_tree, value=1000.0):
    done = 0
    while done < sims:
        n = f.select(trees, per_tree)
        if n == 0:
            break
        f.backup(n, np.full((n, 12), 1 / 12, np.float32), np.full(n, value, np.float32))
        done += n


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    s = Search(threads=threads)
    route = np.load(os.path.join(SEARCH, 'out', 'e2e', 'route.npz'))['actions']
    start = load_state('FullGame')

    # 1. committed play = replay; root stacks = the replayed frames
    f = Forest(s, 1, max_leaves=512)
    assert f.reset(0, start, ROUTE)
    played = []
    for step in range(60):
        think(f, [0], 200, 64)
        visits, best, rb, rn = f.root(0)
        a = int(np.argmax(visits))
        played.append(a)
        assert f.commit(0, a) == Forest.RUNNING
    full, ram, stack = f.state(0)
    obs, tr, end = s.replay_obs(start, np.array(played, np.uint8))
    assert full == end, 'forest root state != replay'
    assert np.array_equal(stack, obs[-4:]), 'root stack != replayed frames'
    print('commit = replay over %d decisions, stack ok, %d nodes kept' % (len(played), f.nodes(0)))

    # 2. forced moves: in control -> not forced; the 1-2 intermediate screen -> forced
    _, s_end11 = s.replay(start, route[:397])
    f2 = Forest(s, 2, max_leaves=64)
    assert f2.reset(0, start, ROUTE) and f2.reset(1, s_end11, ROUTE)
    fz = f2.forced([0, 1])
    print('forced: 1-1 start %s, 1-2 intermediate screen %s (level %s)' % (fz[0], fz[1], s.level(s_end11)))
    assert not fz[0] and fz[1]

    # 3. the goal: 6 steps before 1-2 loads (the castle walk, forced) the search sees it
    #    (every simulation that gets there returns 4 frames per step), and the 6th commit is the goal
    _, s390 = s.replay(start, route[:390])
    f3 = Forest(s, 1, max_leaves=256)
    assert f3.reset(0, s390, ROUTE)
    think(f3, [0], 3000, 128, value=0.0)
    visits, best, rb, rn = f3.root(0)
    terms = [f3.commit(0, 0) for _ in range(6)]
    print('6 steps before the goal: root b = %.1f frames (<= 24), commits %s' % (rb, terms))
    assert 0 < rb <= 24.0 and terms == [0, 0, 0, 0, 0, Forest.GOAL]

    # 4. throughput: one tree, waves of 256
    f4 = Forest(s, 1, max_leaves=256)
    assert f4.reset(0, load_state('FullGame'), ROUTE)
    t = time.time()
    think(f4, [0], 20000, 256)
    dt = time.time() - t
    print('throughput (%d threads, stand-in net): %.0f simulations/s = %.0f per 80 ms'
          % (s.threads, 20000 / dt, 20000 / dt * 0.08))
    print('PASS')


if __name__ == '__main__':
    main()
