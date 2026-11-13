"""Standard 2D U-Net (Ronneberger 2015 layout: 2x conv3x3-BN-ReLU per level, maxpool down, transposed-conv up)
with optional density (1 channel, g/cm3) and class (len(CLASSES) logits) 1x1 heads on a shared decoder."""
import torch
from torch import nn


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=1, base=32, depth=4, density=True, n_cls=0):
        super().__init__()
        ch = [base * 2 ** k for k in range(depth + 1)]
        self.down = nn.ModuleList([block(in_ch, ch[0])] + [block(ch[k - 1], ch[k]) for k in range(1, depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[k], ch[k - 1], 2, stride=2) for k in range(depth, 0, -1)])
        self.dec = nn.ModuleList([block(2 * ch[k - 1], ch[k - 1]) for k in range(depth, 0, -1)])
        self.density = nn.Conv2d(ch[0], 1, 1) if density else None
        self.cls = nn.Conv2d(ch[0], n_cls, 1) if n_cls else None

    def forward(self, x):
        skips = []
        for k, d in enumerate(self.down):
            x = d(x if k == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), s], 1))
        out = {}
        if self.density is not None:
            out["density"] = self.density(x)[:, 0]
        if self.cls is not None:
            out["cls"] = self.cls(x)
        return out
