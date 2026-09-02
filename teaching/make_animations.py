"""
Two BOOST animations for Chapter 1 (eps = 0.05), each tracking the L2 error.

Notation matches the chapter: target p^*, model p_t, tests f in F.

Panels (per frame):
  left   : target p^* and model p_t building up
  middle : indistinguishability error of every test in the current class,
           |E[f*(p_t - p^*)]|, with the selected (argmax) test in orange + eps band
  right  : L2 error  ||p^* - p_t||_2  as a function of the iteration

Animation A  (boost_xonly.gif):   x-only class F = {1, x, x^2, x^3, x cos x}.
Animation B  (boost_binning.gif): the same class augmented from the start with
             the binning tests 1{p(x) in B_k}. The L2 curve shows the extra
             drop the bins buy.

Outputs: boost_xonly.gif / boost_xonly_final.png,
         boost_binning.gif / boost_binning_final.png
The final frames are the figures included in the book (intro/figures/).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

EPS = 0.05
N = 1600
x = np.linspace(-2.5, 2.5, N)
clip = lambda v: np.clip(v, -1.0, 1.0)

trend  = 0.45 * np.sin(1.15 * x)
block1 = 0.65 * ((x >= -1.0) & (x <= 0.9)).astype(float)
block2 = -0.55 * (x > 1.4).astype(float)
block3 = 0.50 * (x < -1.6).astype(float)
g = clip(trend + block1 + block2 + block3 - 0.05)  # the target p^*

def unit(c):
    n = np.sqrt(np.mean(c * c)); return c / n if n > 1e-12 else c
def l2(h):
    return float(np.sqrt(np.mean((g - h) ** 2)))

F0 = [("1", unit(np.ones_like(x))),
      ("x", unit(x)),
      ("x^2", unit(x**2)),
      ("x^3", unit(x**3)),
      ("x cos x", unit(x * np.cos(x)))]

EDGES = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.25), 4)
BUCKETS = list(zip(EDGES[:-1], EDGES[1:]))
LS_LABELS = [f"p:[{lo:+.2f}]" for (lo, hi) in BUCKETS]

def levelset_vec(h, lo, hi):
    ind = ((h >= lo) & (h <= hi)) if hi >= 1.0 else ((h >= lo) & (h < hi))
    ind = ind.astype(float)
    n = np.sqrt(np.mean(ind * ind))
    return ind / n if n > 1e-9 else None

def errors(h, with_ls):
    resid = g - h
    out = [(lab, abs(np.mean(f * resid)), "x") for lab, f in F0]
    if with_ls:
        for (lo, hi), lab in zip(BUCKETS, LS_LABELS):
            v = levelset_vec(h, lo, hi)
            out.append((lab, abs(np.mean(v * resid)) if v is not None else 0.0, "ls"))
    return out

def selected_dir(h, with_ls):
    resid = g - h
    best = (None, 0.0, None)
    for lab, f in F0:
        c = np.mean(f * resid)
        if abs(c) > abs(best[1]): best = (lab, c, f)
    if with_ls:
        for (lo, hi), lab in zip(BUCKETS, LS_LABELS):
            v = levelset_vec(h, lo, hi)
            if v is None: continue
            c = np.mean(v * resid)
            if abs(c) > abs(best[1]): best = (lab, c, v)
    return best

def build_frames(with_ls, class_label):
    """Single-phase boost FROM SCRATCH with a fixed test class.
    with_ls=False -> tests of x only; with_ls=True -> tests of x and p(x)."""
    frames = []
    h = np.zeros_like(g)
    def rec(sel, note):
        frames.append(dict(h=h.copy(), phase=class_label, errs=errors(h, with_ls),
                           sel=sel, note=note, with_ls=with_ls, l2=l2(h)))
    while True:
        lab, c, v = selected_dir(h, with_ls=with_ls)
        if abs(c) < EPS:
            rec(None, "all tests within +-eps"); break
        rec(lab, f"selected: {lab}")
        h = clip(h + EPS * np.sign(c) * v)
    return frames, None

C_G, C_H = "#222222", "#1f77b4"
C_X, C_LS, C_SEL = "#9ecae1", "#c6b0e0", "#e8743b"

def render(frames, aug_idx, outgif, suptitle):
    T = len(frames)
    l2seq = [f["l2"] for f in frames]
    l2max = max(l2seq) * 1.08
    plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.25})
    fig, (axL, axM, axR) = plt.subplots(
        1, 3, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.25, 0.85]})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.24, wspace=0.28)

    def draw(i):
        fr = frames[i]; h = fr["h"]
        axL.clear(); axM.clear(); axR.clear()
        # left
        axL.plot(x, g, color=C_G, lw=2.4, label="target $p^*$")
        axL.plot(x, h, color=C_H, lw=2.2, label="model $p_t$")
        axL.set_ylim(-1.15, 1.15); axL.set_xlabel("$x$"); axL.set_ylabel("value")
        axL.set_title("Model building up", fontweight="bold")
        axL.legend(loc="upper right", framealpha=0.9); axL.grid(alpha=0.25)
        # middle: per-test error
        labels = [e[0] for e in fr["errs"]]; vals = [e[1] for e in fr["errs"]]
        kinds = [e[2] for e in fr["errs"]]; pos = np.arange(len(labels))
        colors = [C_SEL if (fr["sel"] is not None and lab == fr["sel"])
                  else (C_X if k == "x" else C_LS) for lab, k in zip(labels, kinds)]
        axM.axhspan(0, EPS, color="green", alpha=0.10)
        axM.axhline(EPS, color="green", lw=1.0, ls="--", label="$\\varepsilon=0.05$")
        axM.bar(pos, vals, color=colors, edgecolor="#444", linewidth=0.4)
        axM.set_xticks(pos); axM.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        axM.set_ylim(0, 0.25)
        axM.set_ylabel("indistinguishability error  $|E[\\,f\\cdot(p_t-p^*)]|$")
        axM.set_title("Per-test error (orange = selected)", fontweight="bold")
        axM.legend(loc="upper left", framealpha=0.9); axM.grid(alpha=0.25, axis="y")
        if fr["with_ls"]:
            axM.axvline(len(F0) - 0.5, color="#888", lw=1.0, ls=":")
            axM.text(len(F0) - 0.5, 0.242, "  binning tests", fontsize=8, color="#555", va="top")
        # right: L2 error curve
        axR.plot(range(i + 1), l2seq[:i + 1], color="#d6336c", lw=2.0, marker="o", ms=3)
        axR.scatter([i], [l2seq[i]], color="#d6336c", zorder=5)
        if aug_idx is not None and i >= aug_idx:
            axR.axvline(aug_idx, color="#888", lw=1.0, ls=":")
            axR.text(aug_idx, l2max * 0.96, " bins added", fontsize=8, color="#555", va="top")
        axR.set_xlim(-0.5, T - 0.5); axR.set_ylim(0, l2max)
        axR.set_xlabel("iteration $t$"); axR.set_ylabel("$\\|p^*-p_t\\|_2$")
        axR.set_title(f"L2 error = {l2seq[i]:.3f}", fontweight="bold")
        axR.grid(alpha=0.25)

        note = f"   |   {fr['note']}" if fr["note"] else ""
        fig.suptitle(f"{suptitle}    step {i}/{T-1}{note}",
                     fontsize=12.5, fontweight="bold")

    order = []
    for i in range(T):
        order.append(i)
        if aug_idx is not None and i == aug_idx:
            order += [i, i, i]
    order += [T - 1] * 6
    anim = FuncAnimation(fig, draw, frames=order, interval=650)
    anim.save(outgif, writer=PillowWriter(fps=1.6))
    draw(T - 1)
    fig.savefig(outgif.replace(".gif", "_final.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outgif}  (T={T}, final L2={l2seq[-1]:.3f})")

# Animation A: tests of x only (from scratch)
fa, _ = build_frames(with_ls=False, class_label="x-only")
render(fa, None, "boost_xonly.gif", "Distinguishers depend on x only")

# Animation B: tests of x and the model output p(x) (from scratch, full class at step 0)
fb, _ = build_frames(with_ls=True, class_label="x and p(x)")
render(fb, None, "boost_binning.gif", "Distinguishers depend on x and the output p(x)")
