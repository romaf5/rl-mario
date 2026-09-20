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
from verify_retro import verify, retro_state_of
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
from smbsearch import load_state, level_name

FPS = 60.0988


def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def frames_of(start, actions, title, retro_state='Level1-1', scale=3):
    """verified frames with the HUD; returns (list of HxWx3, splits)"""
    out, splits, cur = [], [], [None]
    font, small = _font(7 * scale), _font(5 * scale + 1)

    def on_frame(f, screen, ram):
        lvl = level_name(int(ram[0x75F]) * 4 + int(ram[0x75C]))
        if lvl != cur[0]:
            splits.append((lvl, f)); cur[0] = lvl
        im = Image.fromarray(screen).resize((screen.shape[1] * scale, screen.shape[0] * scale), Image.NEAREST)
        # the HUD sits in its own bar above the game picture (the game's own HUD stays readable)
        bh = 11 * scale
        bar = Image.new('RGB', (im.width, bh), (0, 0, 0))
        d = ImageDraw.Draw(bar)
        d.text((3 * scale, 2 * scale), '%s   %s   frame %d   %.2f s' % (title, lvl, f, f / FPS), fill=(255, 255, 255), font=font)
        sp = '   '.join('%s %.2f' % (l, f0 / FPS) for l, f0 in splits[1:][-3:])
        if sp:
            w = d.textlength(sp, font=small)
            d.text((im.width - 3 * scale - w, 3 * scale), sp, fill=(255, 220, 0), font=small)
        full = Image.new('RGB', (im.width, im.height + bh))
        full.paste(bar, (0, 0)); full.paste(im, (0, bh))
        out.append(np.asarray(full))

    res = verify(start, actions, on_frame=on_frame, retro_state=retro_state)
    if not res['ok']:
        raise SystemExit('[render] verification FAILED at frame %d: %s' % (res['frames'], res['mismatch']))
    return out, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('route')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--scale', type=int, default=3, help='mp4 pixel scale')
    ap.add_argument('--gif-every', type=int, default=3, help='GIF keeps every Nth frame')
    ap.add_argument('--gif-width', type=int, default=240)
    a = ap.parse_args()
    z = np.load(a.route)
    out = a.out or os.path.dirname(os.path.abspath(a.route))
    start = load_state(str(z['start']))
    rs = retro_state_of(str(z['start']))
    fr, splits = frames_of(start, z['actions'], 'optimised', rs, a.scale)
    imageio.mimwrite(os.path.join(out, 'demo.mp4'), fr, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    gh = lambda x: int(round(x.shape[0] * a.gif_width / x.shape[1]))
    gif = [np.asarray(Image.fromarray(x).resize((a.gif_width, gh(x)), Image.BILINEAR)) for x in fr[::a.gif_every]]
    imageio.mimwrite(os.path.join(out, 'demo.gif'), gif, duration=a.gif_every / FPS, loop=0)
    print('[render] demo.mp4: %d frames (%.2f s); splits %s' % (len(fr), len(fr) / FPS,
                                                              ', '.join('%s %.2f' % (l, f / FPS) for l, f in splits)))
    if a.compare and 'ref_actions_0' in z.files:
        lead = np.zeros(int(z['lead_in']), np.uint8)
        rf, _ = frames_of(start, np.concatenate([lead, z['ref_actions_0']]), 'discovered', rs, a.scale)
        n = max(len(rf), len(fr))
        pad = lambda L: L + [L[-1]] * (n - len(L))
        both = [np.concatenate([x, y], axis=1) for x, y in zip(pad(rf), pad(fr))]
        imageio.mimwrite(os.path.join(out, 'compare.mp4'), both, fps=FPS, codec='libx264', quality=7, macro_block_size=1)
        print('[render] compare.mp4: discovered %d vs optimised %d frames' % (len(rf), len(fr)))


if __name__ == '__main__':
    main()
