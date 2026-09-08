"""TTK persistence simplification and EMD tracking for cloud1--cloud3."""
from pathlib import Path
import csv
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import topologytoolkit as ttk
from scipy.ndimage import maximum_filter
import ot

ROOT = Path('/workspace')
DATA = ROOT / 'cloud'
OUT = ROOT / 'cloud_results'
OUT.mkdir(exist_ok=True)
FRAMES = [DATA / f'cloud{i}.vti' for i in (1, 2, 3)]

def read_simplify(path):
    reader = vtk.vtkXMLImageDataReader(); reader.SetFileName(str(path)); reader.Update()
    simp = ttk.ttkTopologicalSimplificationByPersistence()
    simp.SetInputData(reader.GetOutput())
    simp.SetInputArrayToProcess(0, 0, 0, 0, 'Scalars_')
    simp.SetPersistenceThreshold(0.5); simp.SetThresholdIsAbsolute(True); simp.Update()
    data = simp.GetOutput()
    writer = vtk.vtkXMLImageDataWriter(); writer.SetFileName(str(OUT / (path.stem + '_persistence_0.5.vti')))
    writer.SetInputData(data); writer.Write()
    a = vtk_to_numpy(data.GetPointData().GetArray('Scalars_')).astype(float)
    # VTK image points are x-fastest; transpose gives conventional y,x image layout.
    return a.reshape((256, 256)), data

def maxima(field, count=12, separation=9):
    mask = field == maximum_filter(field, size=3, mode='nearest')
    ys, xs = np.nonzero(mask)
    candidates = sorted(zip(field[ys, xs], xs, ys), reverse=True)
    selected = []
    for val, x, y in candidates:
        if all((x-px)**2 + (y-py)**2 >= separation**2 for _, px, py in selected):
            selected.append((float(val), int(x), int(y)))
            if len(selected) == count: break
    return selected

fields, vtk_fields = zip(*(read_simplify(f) for f in FRAMES))
peaks = [maxima(f) for f in fields]

# Earth mover transport between uniformly weighted maxima.  Each positive entry
# of the optimal transport plan produces an EMD-supported PL trajectory edge.
edges = []
for t in range(2):
    a, b = peaks[t], peaks[t+1]
    xy_a = np.array([[p[1], p[2]] for p in a], dtype=float)
    xy_b = np.array([[p[1], p[2]] for p in b], dtype=float)
    cost = ot.dist(xy_a, xy_b, metric='euclidean')
    plan = ot.emd(np.full(len(a), 1/len(a)), np.full(len(b), 1/len(b)), cost)
    for i, j in zip(*np.where(plan > 1e-12)):
        edges.append((t, i, t+1, j, float(plan[i, j]), float(cost[i, j])))

with open(OUT / 'emd_maxima_tracks.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['source_time','source_maximum','target_time','target_maximum','transport_mass','emd_distance'])
    w.writerows(edges)

# Write maxima as actual radius-2 spheres, plus track segments, for ParaView.
points = vtk.vtkPoints(); time_a = vtk.vtkIntArray(); time_a.SetName('TimeStep')
value_a = vtk.vtkFloatArray(); value_a.SetName('ScalarValue')
ids = {}
for t, pp in enumerate(peaks):
    for i, (v,x,y) in enumerate(pp):
        ids[t,i] = points.InsertNextPoint(x,y,2.0)
        time_a.InsertNextValue(t+1); value_a.InsertNextValue(v)
poly = vtk.vtkPolyData(); poly.SetPoints(points); poly.GetPointData().AddArray(time_a); poly.GetPointData().AddArray(value_a)
sphere = vtk.vtkSphereSource(); sphere.SetRadius(2.0); sphere.SetThetaResolution(16); sphere.SetPhiResolution(12)
glyph = vtk.vtkGlyph3D(); glyph.SetInputData(poly); glyph.SetSourceConnection(sphere.GetOutputPort()); glyph.Update()
w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(OUT/'tracked_maxima_spheres_radius_2.vtp')); w.SetInputData(glyph.GetOutput()); w.Write()
lines = vtk.vtkCellArray(); mass = vtk.vtkFloatArray(); mass.SetName('TransportMass')
for t,i,u,j,m,d in edges:
    line = vtk.vtkLine(); line.GetPointIds().SetId(0, ids[t,i]); line.GetPointIds().SetId(1, ids[u,j]); lines.InsertNextCell(line); mass.InsertNextValue(m)
track = vtk.vtkPolyData(); track.SetPoints(points); track.SetLines(lines); track.GetCellData().AddArray(mass)
w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(OUT/'emd_piecewise_linear_tracks.vtp')); w.SetInputData(track); w.Write()

# High-quality static renders (warm--cold/coolwarm field; radius=2 world-unit markers).
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.rcParams.update({'figure.dpi': 160})
fig, ax = plt.subplots(figsize=(7,6)); im=ax.imshow(fields[0], origin='lower', cmap='coolwarm', vmin=0, vmax=50)
ax.set(title='Original scalar field — cloud1', xlabel='x', ylabel='y'); fig.colorbar(im, ax=ax, label='Scalars_'); fig.tight_layout(); fig.savefig(OUT/'original_scalar_field_cloud1.png'); plt.close(fig)
fig, axs = plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
for t, ax in enumerate(axs):
    im=ax.imshow(fields[t],origin='lower',cmap='coolwarm',vmin=0,vmax=50)
    ax.set_title(f'Time step {t+1}: cloud{t+1}')
    ax.set_xlabel('x')
    if t==0: ax.set_ylabel('y')
    for v,x,y in peaks[t]:
        ax.add_patch(Circle((x,y),2,fill=False,edgecolor='#161616',linewidth=1.25))
        ax.plot(x,y,'o',color='#ffd166',markersize=2.5)
    if t < 2:
        # Link only across panels is represented by same peak labels.
        for i,(v,x,y) in enumerate(peaks[t]): ax.text(x+2.5,y+2.5,str(i),fontsize=6,color='black',weight='bold')
        for i,(v,x,y) in enumerate(peaks[t+1]): axs[t+1].text(x+2.5,y+2.5,str(i),fontsize=6,color='black',weight='bold')
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.78,label='Simplified Scalars_ (warm–cold)')
fig.suptitle('Persistence simplification = 0.5; maxima tracked by earth mover transport',y=1.02)
fig.tight_layout(); fig.savefig(OUT/'tracked_maxima_warm_cold.png',bbox_inches='tight'); plt.close(fig)
print('Simplified frames:', ', '.join(str(OUT/(f.stem+'_persistence_0.5.vti')) for f in FRAMES))
print('Maxima per frame:', [len(x) for x in peaks], 'EMD PL edges:', len(edges))
