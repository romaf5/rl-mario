"""Imagined frames: a decoder from the world model's latent to the screen.

The factorization says the latent search's largest lever is the prior below the root: the policy
net on the real screens lifts the all-oracle search from 7 to 11 of 32, and a policy head on the
latent -- distilled from that net -- plateaus at 60-70% agreement. So instead of teaching the latent
to be the net, let the model draw the screen and give the net (and the value net) what they were
trained on. The game is still never stepped to think: every frame past the root is imagined.

Trains a decoder on a frozen world model (its unrolled latents -> the real frame each step led to),
then measures what the search needs: the policy net's top move and the value net's W on imagined
four-frame stacks against the real ones, by depth.

  CUDA_VISIBLE_DEVICES=1 venv_retro/bin/python -m smbzero.wmdec --model smbzero/runs/wm18/wm.pt --out smbzero/runs/dec0
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import DATA, RUNS
from .model import Decoder, load as load_model
from .wmtrain import Trajectories

WORLD = ','.join(os.path.join(DATA, d, '*.npz') for d in ('world', 'world_agent', 'world_sib2', 'world_app'))


def load(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    d = Decoder()
    d.load_state_dict(ck['state'])
    return d.to(device).eval(), ck


def stacks_along(root, dec):
    """root (B, 4, 84, 84) real frames; dec (B, K+1, 84, 84) decoded frames at depths 0..K ->
    the four-frame stack at each depth 1..K: real frames above the root, imagined below it."""
    B, K1 = dec.shape[:2]
    seq = torch.cat([root.float(), dec[:, 1:]], 1)                     # (B, 4 + K, 84, 84)
    return torch.stack([seq[:, k:k + 4] for k in range(1, K1)], 1)    # (B, K, 4, 84, 84)


@torch.no_grad()
def evaluate(model, dec, data, rows, policy, value, device='cuda'):
    """Per depth: the policy net's top move and the value net's W, imagined stack against real."""
    K = data.unroll
    agree = np.zeros(K); cnt = np.zeros(K); werr = np.zeros(K); pix = np.zeros(K)
    a0 = n0 = 0
    for i in range(0, len(rows), 128):
        b = data.batch(rows[i:i + 128], device)
        obs, acts, tgt, valid, prev = b[0], b[1], b[3], b[5], b[8]
        B = len(obs)
        with torch.autocast('cuda', dtype=torch.float16):
            lat = model.unroll(obs, acts, prev)[0]
            d = torch.stack([dec(l) for l in lat], 1).float()               # (B, K+1, 84, 84)
        real = tgt.view(B, K, 4, 84, 84).float()                             # the real stack after each step
        imag = stacks_along(obs, d).round().clamp(0, 255)
        v = valid.bool()
        pix += ((imag[:, :, -1] - real[:, :, -1]).abs().mean((2, 3)) * valid).sum(0).cpu().numpy()
        with torch.autocast('cuda', dtype=torch.float16):
            pr = policy(real.flatten(0, 1).to(torch.uint8))[0].argmax(1).view(B, K)
            pi = policy(imag.flatten(0, 1).to(torch.uint8))[0].argmax(1).view(B, K)
            roots = obs.repeat_interleave(K, 0)
            deps = torch.arange(1, K + 1, device=device).repeat(B)
            wr = value(real.flatten(0, 1).to(torch.uint8), roots, deps).float().view(B, K)
            wi = value(imag.flatten(0, 1).to(torch.uint8), roots, deps).float().view(B, K)
        agree += ((pr == pi) & v).sum(0).cpu().numpy()
        st0 = obs.clone()                  # depth 0: the real screen, encoded and drawn back
        st0[:, -1] = d[:, 0].round().clamp(0, 255).to(torch.uint8)
        with torch.autocast('cuda', dtype=torch.float16):
            a0 += int((policy(obs)[0].argmax(1) == policy(st0)[0].argmax(1)).sum()); n0 += B
        werr += ((wr - wi).abs() * valid).sum(0).cpu().numpy()
        cnt += valid.sum(0).cpu().numpy()
    evaluate.depth0 = a0 / max(n0, 1)
    return agree / cnt, werr / cnt, pix / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='a world model (frozen)')
    ap.add_argument('--data', default=WORLD)
    ap.add_argument('--unroll', type=int, default=12)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--policy', default=os.path.join(RUNS, 'zero8', 'net.pt'))
    ap.add_argument('--value', default=os.path.join(RUNS, 'relv3', 'relvalue.pt'))
    ap.add_argument('--eval', action='store_true', help="score the model's own decoder (trained jointly), no training")
    ap.add_argument('--out', default=os.path.join(RUNS, 'dec0'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'train.log'), 'a')
    log = lambda m: (print(m, flush=True), logf.write(m + '\n'), logf.flush())
    model, _ = load_model(a.model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    from .net import load as load_net
    from .relvalue import load as load_rel
    policy = load_net(a.policy)[0].eval()
    value = load_rel(a.value)[0].cuda().eval()
    data = Trajectories(a.data, a.unroll)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(data))
    va, tr = perm[:2048], perm[2048:]
    if a.eval:
        ag, we, px = evaluate(model, model.dec, data, va, policy, value)
        print('[dec] %s: the policy net keeps its move on imagined stacks -- depth 0 %.0f%%, depth 1-12: %s'
              % (a.model, 100 * evaluate.depth0, ' '.join('%.0f' % (100 * x) for x in ag)))
        print('[dec] value net W, imagined against real, depth 1-12: %s frames; pixel error on the last '
              'frame %s' % (' '.join('%.1f' % x for x in we), ' '.join('%.0f' % x for x in px)))
        return
    dec = Decoder().cuda()
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)
    log('[dec] %s frozen; %d unroll starts' % (a.model, len(data)))
    t0 = time.time()
    K = a.unroll
    for step in range(1, a.steps + 1):
        b = data.batch(rng.choice(tr, a.batch), 'cuda')
        obs, acts, tgt, valid, prev = b[0], b[1], b[3], b[5], b[8]
        B = len(obs)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            lat = model.unroll(obs, acts, prev)[0]
        target = torch.cat([obs[:, -1:].float(), tgt.view(B, K, 4, 84, 84)[:, :, -1].float()], 1)  # (B, K+1, 84, 84)
        m = torch.cat([torch.ones(B, 1, device=obs.device), valid], 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            d = torch.stack([dec(l.float()) for l in lat], 1).float()
        loss = ((d - target).abs().mean((2, 3)) * m).sum() / m.sum() / 255.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == a.steps:
            dec.eval()
            ag, we, px = evaluate(model, dec, data, va, policy, value)
            dec.train()
            log('[dec] step %5d pixel L1 %.1f/255 | imagined against real, depth 1/3/6/12: policy net agrees '
                '%.0f/%.0f/%.0f/%.0f%%, value net W differs %.1f/%.1f/%.1f/%.1f frames (%.0f s)'
                % (step, loss.item() * 255, *(100 * ag[[0, 2, 5, 11]]), *we[[0, 2, 5, 11]], time.time() - t0))
            torch.save(dict(state=dec.state_dict(), step=step, model=a.model, agree=ag.tolist(), werr=we.tolist(),
                            pix=px.tolist()), os.path.join(a.out, 'dec.pt'))


if __name__ == '__main__':
    main()
