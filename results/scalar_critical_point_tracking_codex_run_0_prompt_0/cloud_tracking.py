"""TTK persistence simplification + POT EMD tracking for cloud1..cloud3."""
import os, subprocess
import numpy as np
import vtk
import ot
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = '/workspace'
OUT = os.path.join(ROOT, 'ttk_out')
os.makedirs(OUT, exist_ok=True)
frames, maxima = [], []

def read_vti(path):
    r = vtk.vtkXMLImageDataReader(); r.SetFileName(path); r.Update()
    d = r.GetOutput(); a = d.GetPointData().GetArray('Scalars_')
    x, y, _ = d.GetDimensions()
    return np.array([a.GetTuple1(i) for i in range(a.GetNumberOfTuples())]).reshape(y, x)

def ttk_maxima(frame):
    inp = os.path.join(ROOT, 'cloud', f'cloud{frame}.vti')
    prefix = os.path.join(OUT, f'cloud{frame}')
    # TTK computes the PL persistence diagram; retaining only persistence >= 0.5
    # is the requested persistence simplification of saddle--maximum pairs.
    subprocess.run(['/opt/conda/bin/ttkPersistenceDiagramCmd', '-i', inp,
                    '-a', 'Scalars_', '-o', prefix], check=True,
                   stdout=subprocess.DEVNULL)
    r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(prefix + '_port_0.vtu'); r.Update()
    d = r.GetOutput(); cd, pd = d.GetCellData(), d.GetPointData()
    result = []
    for i in range(d.GetNumberOfCells()):
        if cd.GetArray('PairType').GetTuple1(i) != 1 or cd.GetArray('Persistence').GetTuple1(i) < .5:
            continue
        # The second endpoint of each type-1 pair is the PL maximum. Coordinates
        # holds its original spatial position (not the diagram coordinate).
        pid = d.GetCell(i).GetPointId(1)
        xy = pd.GetArray('Coordinates').GetTuple(pid)
        result.append((xy[0], xy[1], cd.GetArray('Persistence').GetTuple1(i)))
    return np.asarray(result, dtype=float)

for t in (1, 2, 3):
    frames.append(read_vti(os.path.join(ROOT, 'cloud', f'cloud{t}.vti')))
    maxima.append(ttk_maxima(t))

# Exact discrete earth-mover transport between consecutive maxima sets.  The
# maximum entry in each transport row defines that maximum's temporal match.
tracks = []
for a, b in zip(maxima[:-1], maxima[1:]):
    cost = ot.dist(a[:, :2], b[:, :2], metric='euclidean')
    plan = ot.emd(np.full(len(a), 1.0 / len(a)), np.full(len(b), 1.0 / len(b)), cost)
    tracks.append(np.argmax(plan, axis=1))

# Persist all simplified critical points and EMD correspondences in a compact CSV.
with open(os.path.join(ROOT, 'tracked_critical_points.csv'), 'w') as f:
    f.write('time,x,y,persistence,emd_match_next\n')
    for t, pts in enumerate(maxima):
        nxt = tracks[t] if t < 2 else np.full(len(pts), -1)
        for p, m in zip(pts, nxt): f.write(f'{t+1},{p[0]},{p[1]},{p[2]},{int(m)}\n')

# Visualize original scalar fields with warm-cold colours. Circles have a data
# radius of 2, representing the requested critical-point spheres in this 2-D field.
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
vmin, vmax = min(x.min() for x in frames), max(x.max() for x in frames)
for t, ax in enumerate(axes):
    im = ax.imshow(frames[t], origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    pts = maxima[t]
    # all preserved maxima are present; strongest ones receive outlined sphere glyphs
    ax.scatter(pts[:,0], pts[:,1], s=3, c='#ffdf35', alpha=.45, linewidths=0)
    for x, y, _ in pts[np.argsort(pts[:,2])[-100:]]:
        ax.add_patch(Circle((x,y), 2, fill=False, edgecolor='#fff3a0', linewidth=.7))
    ax.set_title(f'Time step {t+1}: {len(pts)} simplified PL maxima')
    ax.set_aspect('equal'); ax.set_xlim(0,255); ax.set_ylim(0,255)
fig.colorbar(im, ax=axes, label='Original Scalars_')
fig.suptitle('Cloud scalar field with persistence ≥ 0.5 maxima (EMD tracked)', fontsize=15)
fig.savefig(os.path.join(ROOT, 'cloud_tracking.png'), dpi=180)
print('maxima:', [len(p) for p in maxima])
