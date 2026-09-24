"""The SMBZero network: 4 x 84 x 84 frames -> prior over the 12 actions, frames to go.

IMPALA CNN (3 stages of conv + max-pool + 2 residual blocks: 16, 32, 32 channels),
FC 256, a policy head (logits) and a value head (sigmoid x V_SCALE frames).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import V_SCALE


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.relu(self.a(F.relu(x))))


class _Stage(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.conv = nn.Conv2d(cin, c, 3, padding=1)
        self.r1, self.r2 = _Res(c), _Res(c)

    def forward(self, x):
        return self.r2(self.r1(F.max_pool2d(self.conv(x), 3, 2, 1)))


class Net(nn.Module):
    def __init__(self, channels=(16, 32, 32), hidden=256):
        super().__init__()
        stages, cin = [], 4
        for c in channels:
            stages.append(_Stage(cin, c)); cin = c
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(cin * 11 * 11, hidden)
        self.pi = nn.Linear(hidden, 12)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        """x: (B, 4, 84, 84) uint8 or float in [0, 255] -> (logits (B, 12), value in [0, 1] (B,))"""
        h = self.stages(x.float() / 255.0)
        h = F.relu(self.fc(F.relu(h).flatten(1)))
        return self.pi(h), torch.sigmoid(self.v(h)).squeeze(1)


class Evaluator:
    """Batched inference for the MCTS forest: reads forest.stacks (pinned), returns
    priors (n, 12) float32 and values (n,) in frames."""
    def __init__(self, net, max_leaves, device='cuda'):
        self.net = net.to(device).eval()
        self.device = device
        self.pinned = torch.zeros((max_leaves, 4, 84, 84), dtype=torch.uint8).pin_memory()
        self.stacks = self.pinned.numpy()             # hand this to Forest(stacks=...)

    @torch.no_grad()
    def __call__(self, n, forest=None):
        x = self.pinned[:n].to(self.device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits, v = self.net(x)
        p = torch.softmax(logits.float(), 1)
        return p.cpu().numpy(), (v.float() * V_SCALE).cpu().numpy()


class RelEvaluator(Evaluator):
    """Prior from the policy net, value from the relative-value net: W frames wasted against
    perfect play from the tree's root, returned as D = W - 4 depth (Forest(relative=True)).
    The root's frames are embedded once per decision."""
    def __init__(self, net, relnet, max_leaves, device='cuda'):
        super().__init__(net, max_leaves, device)
        self.rel = relnet.to(device).eval()
        self.root_e = {}

    @torch.no_grad()
    def new_decision(self, forest, trees):
        st = np.stack([forest.state(t)[2] for t in trees])
        with torch.autocast('cuda', dtype=torch.float16):
            e = self.rel.embed(torch.from_numpy(st).to(self.device))
        self.root_e = {int(t): e[i] for i, t in enumerate(trees)}

    @torch.no_grad()
    def __call__(self, n, forest=None):
        x = self.pinned[:n].to(self.device, non_blocking=True)
        depth = forest.leaf_info(n)[1]
        trees = forest.leaves[:n, 0]
        with torch.autocast('cuda', dtype=torch.float16):
            logits, _ = self.net(x)
            e_root = torch.stack([self.root_e[int(t)] for t in trees])
            w = self.rel.head(self.rel.embed(x), e_root, torch.from_numpy(depth).to(self.device))
        p = torch.softmax(logits.float(), 1)
        return p.cpu().numpy(), w.float().cpu().numpy() - 4.0 * depth


def save(net, path, **meta):
    torch.save(dict(state=net.state_dict(), **meta), path)


def load(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = Net()
    net.load_state_dict(ck['state'])
    return net.to(device), ck
