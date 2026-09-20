"""The net's frames from the search emulator (ss_replay_obs) are batchenv's pixels.

  venv_retro/bin/python search/tests/obs_test.py
"""
import ctypes, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'python'))
from smbsearch import Search, load_state, ROM, REPO, ACTION_BUTTONS


def batchenv_obs(state, actions):
    L = ctypes.CDLL(os.path.join(REPO, 'native', 'libbatchenv.so'))
    P, I = ctypes.c_void_p, ctypes.c_int
    L.benv_create.restype = P; L.benv_create.argtypes = [ctypes.c_char_p, I, I, I, I]
    L.benv_load.argtypes = [P, I, ctypes.c_char_p]
    L.benv_step_raw.argtypes = [P, I, I, P, P]
    L.benv_destroy.argtypes = [P]
    rom = open(ROM, 'rb').read()
    env = L.benv_create(rom, len(rom), 1, 1, 0)
    L.benv_load(env, 0, state)
    obs = np.zeros((len(actions), 84, 84), np.uint8)
    ram = np.zeros(0x800, np.uint8)
    for i, a in enumerate(actions):
        L.benv_step_raw(env, 0, ACTION_BUTTONS[a], obs[i].ctypes.data, ram.ctypes.data)
    L.benv_destroy(env)
    return obs


def main():
    s = Search(threads=4)
    rng = np.random.default_rng(0)
    for name in ('FullGame', '4-2', '8-4'):
        st = load_state(name)
        acts = rng.choice([1, 2, 3, 4, 4, 4, 0, 5], size=600).astype(np.uint8)
        mine, tr, _ = s.replay_obs(st, acts)
        ref = batchenv_obs(st, acts)
        diff = int((mine != ref).sum())
        print('%-8s %d steps: %d differing pixels, mean %.1f' % (name, len(acts), diff, mine.mean()))
        assert diff == 0
    st = load_state('4-2')
    acts = rng.integers(0, 12, 20000).astype(np.uint8)
    t = time.time(); s.replay(st, acts); t1 = time.time() - t
    t = time.time(); s.replay_obs(st, acts); t2 = time.time() - t
    print('one thread: step %.1f us, step+frame %.1f us (%.1fx)' % (t1 / len(acts) * 1e6, t2 / len(acts) * 1e6, t2 / t1))
    print('PASS')


if __name__ == '__main__':
    main()
