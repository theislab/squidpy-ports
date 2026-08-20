"""Extract just what a figure needs from the big npz files, plus scores against the section."""
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
keep, J = {}, None
for p in sorted(root.glob("*.npz")):
    if p.stem in {"panels", "figure-inputs"}:
        continue
    d = np.load(p)
    if J is None:
        J = d["section"]
    keep[f"{p.stem}__panel"] = d["panel"].astype(np.float32)
    keep[f"{p.stem}__ai0"] = d["AI"][0, 0].astype(np.float32)
    keep[f"{p.stem}__A"] = d["A"]
    keep[f"{p.stem}__energies"] = d["energies"].astype(np.float32)
keep["section"] = J.astype(np.float32)

mask = J > 0.05 * J.max()
def score(a):
    x, y = a[mask] - a[mask].mean(), J[mask] - J[mask].mean()
    corr = float((x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum()))
    design = np.stack([a[mask], np.ones(int(mask.sum()))], 1)
    coef, *_ = np.linalg.lstsq(design, J[mask], rcond=None)
    return corr, float(np.sqrt(((design @ coef - J[mask]) ** 2).mean()))

names = sorted({k.split("__")[0] for k in keep if "__" in k})
print(f"{'condition':16s} {'corr':>8s} {'residual':>9s}")
for n in names:
    c, r = score(keep[f"{n}__ai0"])
    keep[f"{n}__corr"], keep[f"{n}__resid"] = c, r
    print(f"{n:16s} {c:+8.4f} {r:9.5f}")

if "upstream-rep1" in names and "upstream-rep2" in names:
    dc = abs(keep["upstream-rep1__corr"] - keep["upstream-rep2__corr"])
    dr = abs(keep["upstream-rep1__resid"] - keep["upstream-rep2__resid"])
    pd = float(np.sqrt(((keep["upstream-rep1__panel"] - keep["upstream-rep2__panel"]) ** 2).mean()))
    keep["upstream_corr_floor"], keep["upstream_resid_floor"], keep["upstream_panel_floor"] = dc, dr, pd
    print(f"\nupstream's OWN rep-to-rep wander (the floor for anything measured against it):")
    print(f"  corr {dc:.4f}   residual {dr:.5f}   panel rms {pd:.4f}")
    for a, b in (("3axis-base", "2axis-base"),):
        gap = abs(keep[f"{a}__corr"] - keep[f"{b}__corr"])
        print(f"  {a} vs {b} corr gap {gap:.4f} -> "
              f"{gap/dc:.1f}x that floor" if dc else "  floor is zero")
np.savez_compressed(root / "panels.npz", **keep)
print("\nwrote", root / "panels.npz")
