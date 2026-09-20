"""Render a verified route as a demo: 60 fps mp4 (3x, HUD with level, frame
counter and per-level splits) and a smaller GIF; --compare adds a side-by-side
of the explore reference vs the optimised route for single-segment routes.

  venv_retro/bin/python search/tools/render_demo.py search/out/4-2/route.npz --compare
"""
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_retro import verify
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
from smbsearch import load_state, level_name

FPS = 60.0988


def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def frames_of(start, actions, title):
    """verified frames with the HUD; returns (list of HxWx3, splits)"""
    out, splits, cur = [], [], [None]
    font, small = _font(20), _font(16)

    def on_frame(f, screen, ram):
        lvl = level_name(int(ram[0x75F]) * 4 + int(ram[0x75C]))
        if lvl != cur[0]:
            splits.append((lvl, f)); cur[0] = lvl
        im = Image.fromarray(screen).resize((screen.shape[1] * 3, screen.shape[0] * 3), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
        d.text((8, 4), '%s   %s   frame %d   %.2f s' % (title, lvl, f, f / FPS), fill=(255, 255, 255), font=font)
        for j, (l, f0) in enumerate(splits[-6:]):
            d.text((im.width - 180, 40 + 20 * j), '%s @ %.2f s' % (l, f0 / FPS), fill=(255, 255, 0), font=small)
        out.append(np.asarray(im))

    res = verify(start, actions, on_frame=on_frame)
    if not res['ok']:
        raise SystemExit('[render] verification FAILED at frame %d: %s' % (res['frames'], res['mismatch']))
    return out, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('route')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    z = np.load(a.route)
    out = a.out or os.path.dirname(os.path.abspath(a.route))
    start = load_state(str(z['start']))
    fr, splits = frames_of(start, z['actions'], 'optimised')
    imageio.mimwrite(os.path.join(out, 'demo.mp4'), fr, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    gif = [np.asarray(Image.fromarray(x).resize((x.shape[1] // 3, x.shape[0] // 3))) for x in fr[::3]]
    imageio.mimwrite(os.path.join(out, 'demo.gif'), gif, duration=3 / FPS, loop=0)
    print('[render] demo.mp4: %d frames (%.2f s); splits %s' % (len(fr), len(fr) / FPS,
                                                              ', '.join('%s %.2f' % (l, f / FPS) for l, f in splits)))
    if a.compare and 'ref_actions_0' in z.files:
        lead = np.zeros(int(z['lead_in']), np.uint8)
        rf, _ = frames_of(start, np.concatenate([lead, z['ref_actions_0']]), 'discovered')
        n = max(len(rf), len(fr))
        pad = lambda L: L + [L[-1]] * (n - len(L))
        both = [np.concatenate([x, y], axis=1) for x, y in zip(pad(rf), pad(fr))]
        imageio.mimwrite(os.path.join(out, 'compare.mp4'), both, fps=FPS, codec='libx264', quality=7, macro_block_size=1)
        print('[render] compare.mp4: discovered %d vs optimised %d frames' % (len(rf), len(fr)))


if __name__ == '__main__':
    main()
