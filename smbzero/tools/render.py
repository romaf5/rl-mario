"""Render an SMBZero game (eval route file: start, lead_frames, actions) as a 4x, 50 fps mp4,
replayed in stable-retro (every frame checked). A level game starts at that level's entry
state from the e2e route; --until-level cuts a full game N decisions into a level.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.tools.render GAME.npz OUT.mp4 [--until-level 8-1 --after 150]
"""
import argparse, os, sys
import numpy as np
from ..common import REPO, Search, e2e_segments, gp, load_state
sys.path.insert(0, os.path.join(REPO, 'search', 'tools'))
import imageio.v2 as imageio
from render_demo import frames_of, FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game'); ap.add_argument('out')
    ap.add_argument('--until-level'); ap.add_argument('--after', type=int, default=150)
    ap.add_argument('--title', default='SMBZero')
    a = ap.parse_args()
    z = np.load(a.game)
    s = Search(threads=4)
    name, lead, acts = str(z['start']), int(z['lead_frames']), z['actions']
    start = load_state('FullGame') if name == 'FullGame' else {g['level']: g for g in e2e_segments(s)}[name]['start']
    if a.until_level:                               # cut the game a little way into that level
        tr, _ = s.replay(s.frames(start, lead), acts)
        i = int(np.argmax(tr[:, 2] == gp(a.until_level)))
        acts = acts[:i + a.after]
    w = imageio.get_writer(a.out, fps=FPS, codec='libx264', quality=None, macro_block_size=1,
                           ffmpeg_params=['-preset', 'veryslow', '-tune', 'animation', '-crf', '18', '-threads', '2',
                                          '-movflags', '+faststart'])
    _, splits = frames_of(start, acts, a.title, 'FullGame', 4, w.append_data, lead)
    w.close()
    print('[render] %s: %d decisions (+%d lead frames), %.1f MB, splits %s' % (
        a.out, len(acts), lead, os.path.getsize(a.out) / 1e6, ', '.join('%s %.1f' % (l, f / FPS) for l, f in splits)))


if __name__ == '__main__':
    main()
