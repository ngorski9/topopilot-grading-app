import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import vtk
from vtk.util.numpy_support import vtk_to_numpy

SPHERE_RADIUS = 2.0
CMAP = 'coolwarm'  # warm-cold colormap

def read_vti(fname):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(fname)
    r.Update()
    img = r.GetOutput()
    dims = img.GetDimensions()
    arr = vtk_to_numpy(img.GetPointData().GetArray('Scalars_'))
    field = arr.reshape(dims[1], dims[0])  # y, x
    return field

field0 = read_vti('/workspace/cloud/cloud1.vti')
trajectories = np.load('/workspace/trajectories.npy')  # (n_traj, n_steps, 3)
pers0 = np.load('/workspace/maxima_pers_t0.npy')

# for legibility, display the most persistent tracked maxima (still all pass the 0.5 threshold)
TOP_K = 60
top_idx = np.argsort(pers0)[::-1][:TOP_K]
trajectories = trajectories[top_idx]

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(field0, origin='lower', cmap=CMAP, extent=[0, field0.shape[1], 0, field0.shape[0]])
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('Scalars_')

# Tracked trajectory lines (piecewise-linear paths of tracked maxima across the 3 timesteps)
segments = trajectories[:, :, :2]
lc = LineCollection(segments, colors='black', linewidths=0.6, alpha=0.6, zorder=2)
ax.add_collection(lc)

# Tracked critical points rendered as "spheres" (circles scaled by the requested radius)
colors_by_step = ['#2c7bb6', '#ffffbf', '#d7191c']  # cool -> warm across the 3 timesteps
for t in range(trajectories.shape[1]):
    pts = trajectories[:, t, :2]
    ax.scatter(pts[:, 0], pts[:, 1], s=(SPHERE_RADIUS * 6) ** 2 * 0.02,
               facecolors=colors_by_step[t], edgecolors='black', linewidths=0.3,
               alpha=0.85, zorder=3, label=f'tracked maxima @ t={t}')

ax.set_title('Tracked PL maxima (persistence threshold=0.5) via Earth Mover\'s Distance\nover 3 timesteps, sphere radius=2, warm-cold colormap')
ax.set_xlim(0, field0.shape[1])
ax.set_ylim(0, field0.shape[0])
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig('/workspace/cloud_tracked_maxima.png', dpi=150)
print('Saved /workspace/cloud_tracked_maxima.png')
