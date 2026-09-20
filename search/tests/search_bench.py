"""smbsearch checks. Run: venv_retro/bin/python search/tests/search_bench.py [--level42]

  * compact states: stepping through a compact save/load equals stepping straight through
  * the engine's stepping equals the native core's smb_frame (RAM identical after every action)
  * the button table equals the training env's COMPLEX_MOVEMENT bytes
  * explore finds the 1-1 flag; optimise is no slower than its reference and replays to 1-2
  * (--level42) 4-2: explore + optimise reach 8-1
"""
import ctypes, os, sys, time
import numpy as np
SEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SEARCH, 'python'))
from smbsearch import Search, load_state, ROUTE, gp, ROM, REPO, ACTION_BUTTONS

OK = []


def check(name, cond, detail=''):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name + ('' if cond else '   <- ' + str(detail)), flush=True)


s = Search(threads=int(os.environ.get('SS_THREADS', '32')))
st42, st11 = load_state('4-2'), load_state('1-1')
rs = np.random.RandomState(0)

# ---------------------------------------------------------------- engine
check('compact states: save/load mid-run changes nothing (400 random steps)', s.selftest(st42, 400))
acts = rs.randint(0, 12, 120).astype(np.uint8)
tr, end = s.replay(st42, acts)
lib = ctypes.CDLL(os.path.join(REPO, 'native', 'libsmbcore.so'))
lib.smb_create.restype = ctypes.c_void_p; lib.smb_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.smb_frame.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.smb_ram.restype = ctypes.POINTER(ctypes.c_uint8); lib.smb_ram.argtypes = [ctypes.c_void_p]
lib.smb_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
rom = open(ROM, 'rb').read()
ca, cb = lib.smb_create(rom, len(rom)), lib.smb_create(rom, len(rom))
lib.smb_load(ca, st42)
for a in acts:
    for _ in range(4):
        lib.smb_frame(ca, int(ACTION_BUTTONS[a]))
lib.smb_load(cb, end)
ra = np.ctypeslib.as_array(lib.smb_ram(ca), shape=(0x800,)).copy()
rb = np.ctypeslib.as_array(lib.smb_ram(cb), shape=(0x800,)).copy()
check('stepping: 120 actions equal smb_frame x 4 (RAM identical)', np.array_equal(ra, rb), np.nonzero(ra != rb)[0][:8])
sys.path.insert(0, REPO)
from mario_native_vecenv import _ACTION_BYTES
check('actions: button table equals the training env', list(_ACTION_BYTES) == list(ACTION_BUTTONS))
fps = s.bench(st42, 2_000_000)
check('bench: %.0f frames/s on %d threads' % (fps, s.threads), fps > 1e5, fps)

# ---------------------------------------------------------------- explore / optimise 1-1
t = time.time()
ref = s.explore(st11, ROUTE, budget_s=90, settle_s=15, seed=1)
check('explore 1-1: flag found (%d actions, %d cells, %d walks, %.0f s)'
      % (len(ref.actions), ref.stats['cells'], ref.stats['walks'], time.time() - t), ref.found)
tr, _ = s.replay(st11, ref.actions)
check('explore 1-1: its actions replay into 1-2 at the last step', ref.found and tr[-1, 2] == gp('1-2') and (tr[:-1, 2] == gp('1-1')).all())
t = time.time()
opt = s.optimize(st11, ROUTE, ref.actions, beam=4000, per_cell=16)
check('optimise 1-1: %d actions vs reference %d (%.0f s)' % (len(opt.actions), len(ref.actions), time.time() - t),
      opt.found and len(opt.actions) <= len(ref.actions))
tr, _ = s.replay(st11, opt.actions)
check('optimise 1-1: replays into 1-2 at the last step', opt.found and tr[-1, 2] == gp('1-2') and (tr[:-1, 2] == gp('1-1')).all())

if '--level42' in sys.argv:
    ref = s.explore(st42, ROUTE, budget_s=600, settle_s=120, seed=1)
    check('explore 4-2: route into 8-1 found (%d actions)' % len(ref.actions), ref.found)
    opt = s.optimize(st42, ROUTE, ref.actions, beam=20000, per_cell=16, verbose=1)
    tr, _ = s.replay(st42, opt.actions)
    check('optimise 4-2: %d actions vs %d, replays into 8-1' % (len(opt.actions), len(ref.actions)),
          opt.found and tr[-1, 2] == gp('8-1'))

print('\n%d/%d checks passed' % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
