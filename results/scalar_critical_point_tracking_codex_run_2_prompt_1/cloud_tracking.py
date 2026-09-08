"""Three-step cloud maxima tracking: persistence threshold 0.5 + EMD matching."""
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d

ROOT = Path("/workspace")
FILES = [ROOT / "cloud" / f"cloud{i}.vti" for i in (1, 2, 3)]
THRESHOLD, RADIUS = 0.5, 2.0

def read_field(path):
    r = vtk.vtkXMLImageDataReader(); r.SetFileName(str(path)); r.Update()
    img = r.GetOutput(); dims = img.GetDimensions()
    a = vtk_to_numpy(img.GetPointData().GetArray("Scalars_"))
    # VTK's x-fastest point ordering -> conventional image rows, columns.
    return a.reshape(dims[1], dims[0]), dims

def persistence_maxima(field, threshold):
    """0-D superlevel persistence via union-find; keep maxima >= threshold.

    This is the maxima-side operation of persistence simplification: every
    maximum paired with a merge saddle below `threshold` is removed.
    """
    h, w = field.shape; n = h * w
    parent = np.full(n, -1, int); birth = np.empty(n); peak = np.empty(n, int)
    active = np.zeros(n, bool); kept = []
    order = np.argsort(field.ravel())[::-1]
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for idx in order:
        active[idx] = True; parent[idx] = idx; birth[idx] = field.ravel()[idx]; peak[idx] = idx
        y, x = divmod(idx, w); roots = set()
        for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            j = yy*w+xx
            if 0 <= yy < h and 0 <= xx < w and active[j]: roots.add(find(j))
        for root in roots:
            cur = find(idx); other = find(root)
            if cur == other: continue
            # older component is the one born at the higher maximum
            if birth[cur] < birth[other]: cur, other = other, cur
            persistence = birth[other] - field.ravel()[idx]
            if persistence >= threshold:
                py, px = divmod(peak[other], w)
                kept.append((px, py, float(birth[other]), float(persistence)))
            parent[other] = cur
    # The unpaired global maximum is retained.
    root = find(order[0]); py, px = divmod(peak[root], w)
    kept.append((px, py, float(birth[root]), float(field.max() - field.min())))
    # de-duplicate plateaus and cap only exact duplicates
    result = {}
    for item in kept: result[(item[0], item[1])] = item
    return np.array(list(result.values()), float)

fields = [read_field(f)[0] for f in FILES]
maxima = [persistence_maxima(a, THRESHOLD) for a in fields]
# Cloud fields contain a large number of equal-value raster plateaus.  Retain
# the most persistent representatives for transport so EMD remains a meaningful
# feature-level correspondence rather than a pixel-level transport problem.
MAX_TRACKED_FEATURES = 100
maxima = [m[np.argsort(m[:, 3])[::-1][:MAX_TRACKED_FEATURES]] for m in maxima]

# EMD creates a transport plan at each adjacent pair.  The highest transported
# mass target is the continuation of a source maximum, producing PL segments.
links = []
for a, b in zip(maxima[:-1], maxima[1:]):
    xy_a, xy_b = a[:, :2], b[:, :2]
    cost = ot.dist(xy_a, xy_b, metric="euclidean")
    cost /= max(float(cost.max()), 1.0)
    plan = ot.emd(np.ones(len(a)) / len(a), np.ones(len(b)) / len(b), cost)
    links.append(np.argmax(plan, axis=1))

# Original scalar field: rendered before any simplification/feature overlay.
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
im = ax.imshow(fields[0], origin="lower", cmap="coolwarm", interpolation="nearest")
fig.colorbar(im, ax=ax, label="Scalars_")
ax.set(title="Original scalar field — cloud1", xlabel="x", ylabel="y", aspect="equal")
fig.savefig(ROOT / "cloud_original_scalar.png", dpi=180)
plt.close(fig)

# A three-panel scalar rendering with radius-2 spherical glyph projections and
# EMD-derived piecewise-linear paths.  Glyph outlines are circles of radius 2
# in the dataset's coordinate system, matching the requested sphere radius.
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, constrained_layout=True)
vmin, vmax = min(x.min() for x in fields), max(x.max() for x in fields)
for t, ax in enumerate(axes):
    ax.imshow(fields[t], origin="lower", cmap="coolwarm", interpolation="nearest", vmin=vmin, vmax=vmax)
    for x, y, value, persistence in maxima[t]:
        ax.add_patch(Circle((x, y), RADIUS, fill=False, edgecolor="#111111", linewidth=1.25))
        ax.plot(x, y, "o", color="#ffdf4f", markeredgecolor="#111111", markersize=3.8)
    if t < 2:
        for i, j in enumerate(links[t]):
            p, q = maxima[t][i, :2], maxima[t+1][j, :2]
            ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color="#f4f4f4", lw=0.65, alpha=.75))
    ax.set_title(f"t = {t}  ({len(maxima[t])} retained maxima)")
    ax.set_aspect("equal"); ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap="coolwarm"), ax=axes, label="Scalars_")
fig.suptitle("Persistence-simplified maxima (0.5), EMD piecewise-linear tracking; glyph radius = 2", fontsize=13)
fig.savefig(ROOT / "cloud_tracked_maxima.png", dpi=180)
plt.close(fig)

np.savez(ROOT / "cloud_tracking_results.npz", maxima_t0=maxima[0], maxima_t1=maxima[1], maxima_t2=maxima[2], links_01=links[0], links_12=links[1])
print("retained maxima:", [len(m) for m in maxima])
print("wrote cloud_original_scalar.png, cloud_tracked_maxima.png, cloud_tracking_results.npz")
