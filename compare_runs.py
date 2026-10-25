import sys, io, re, zipfile, torch
from pathlib import Path
from collections import defaultdict

a, b = Path(sys.argv[1]), Path(sys.argv[2])

def step(p):
    m = re.findall(r"\d+", p.stem)
    return int(m[-1]) if m else -1

def load(p):
    with zipfile.ZipFile(p) as z:
        return torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")

groups = defaultdict(list)
for p in a.rglob("*.zip"):
    key = (str(p.parent.relative_to(a)), re.sub(r"\d+", "", p.stem))
    groups[key].append(p)

for (d, name), zs in sorted(groups.items()):
    fa = sorted(zs, key=step)[-1]
    fb = b / d / fa.name
    if not fb.exists():
        print(d, fa.name, "MISSING in B")
        continue
    sa, sb = load(fa), load(fb)
    diff = max((sa[k] - sb[k]).abs().max().item() for k in sa)
    print(f"{d:30s} {fa.name:36s} max|dW| = {diff:.6f}")