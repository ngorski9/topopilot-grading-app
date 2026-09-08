"""TTK + POT visualization of three cloud time steps.

Run with: /opt/conda/bin/pvpython visualization/cloud_tracking.py
Outputs a ParaView state, tracked geometry, and a PNG preview.
"""
from pathlib import Path
import numpy as np
import ot
from scipy.ndimage import maximum_filter
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from paraview.simple import *
from paraview import servermanager

ROOT = Path('/workspace')
OUT = ROOT / 'visualization'
OUT.mkdir(exist_ok=True)
FILES = [ROOT / 'cloud' / f'cloud{i}.vti' for i in (1, 2, 3)]
THRESHOLD = 0.5
N_MAXIMA = 40                 # readable subset after simplification
Z_STEP = 8.0

def simplified_maxima(path):
    """Simplify using TTK, then return the strongest PL maxima."""
    reader = XMLImageDataReader(FileName=[str(path)])
    simp = TTKTopologicalSimplificationByPersistence(Input=reader)
    simp.InputArray = ['POINTS', 'Scalars_']
    simp.PairType = 2          # maximum--saddle pairs
    simp.PersistenceThreshold = THRESHOLD
    simp.ThresholdIsAbsolute = 1
    simp.UpdatePipeline()
    image = servermanager.Fetch(simp)
    arr = vtk_to_numpy(image.GetPointData().GetArray('Scalars_'))
    dims = image.GetDimensions()
    field = arr.reshape((dims[1], dims[0]))
    # Pixel-grid local maxima are piecewise-linear vertex maxima on this 2D grid.
    mask = (field == maximum_filter(field, size=3, mode='nearest'))
    yy, xx = np.nonzero(mask)
    values = field[yy, xx]
    order = np.lexsort((xx, yy, -values))[:N_MAXIMA]
    return np.c_[xx[order], yy[order], np.zeros(len(order))], values[order], reader

points_by_time, values_by_time, readers = zip(*(simplified_maxima(p) for p in FILES))

# EMD (Wasserstein-1) between consecutive point sets.  Equal masses make this
# a discrete earth-mover problem; dominant transport links form trajectories.
links = []
for t in range(2):
    a, b = points_by_time[t][:, :2], points_by_time[t + 1][:, :2]
    cost = ot.dist(a, b, metric='euclidean')
    plan = ot.emd(np.full(len(a), 1.0 / len(a)),
                  np.full(len(b), 1.0 / len(b)), cost)
    for i, j in enumerate(np.argmax(plan, axis=1)):
        if plan[i, j] > 0:
            links.append((t, i, t + 1, int(j)))

# Write one geometry containing time-shifted critical points and EMD links.
vtkpts = vtkPoints(); verts = vtkCellArray(); lines = vtkCellArray(); ids = []
for t, pts in enumerate(points_by_time):
    row = []
    for p in pts:
        row.append(vtkpts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2] + t * Z_STEP)))
    ids.append(row)
for row in ids:
    for pid in row:
        verts.InsertNextCell(1); verts.InsertCellPoint(pid)
for t0, i, t1, j in links:
    lines.InsertNextCell(2); lines.InsertCellPoint(ids[t0][i]); lines.InsertCellPoint(ids[t1][j])
poly = vtkPolyData(); poly.SetPoints(vtkpts); poly.SetVerts(verts); poly.SetLines(lines)
writer = vtkXMLPolyDataWriter(); writer.SetFileName(str(OUT / 'emd_tracked_maxima.vtp')); writer.SetInputData(poly); writer.Write()

# Render original scalar fields in three stacked time planes, with the tracking
# geometry overlaid as radius-2 spheres and semi-transparent EMD trajectories.
view = CreateView('RenderView')
view.ViewSize = [1200, 900]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0
for t, reader in enumerate(readers):
    move = Transform(Input=reader); move.Transform.Translate = [0, 0, t * Z_STEP]
    display = Show(move, view)
    ColorBy(display, ('POINTS', 'Scalars_'))
    display.Representation = 'Surface'
    lut = GetColorTransferFunction('Scalars_')
    lut.RGBPoints = [0.0, 0.230, 0.299, 0.754, 25.0, 0.865, 0.865, 0.865, 50.0, 0.706, 0.016, 0.150]
    lut.ColorSpace = 'Diverging'
    display.LookupTable = lut
    display.SetScalarBarVisibility(view, t == 0)

tracked = XMLPolyDataReader(FileName=[str(OUT / 'emd_tracked_maxima.vtp')])
glyphs = TTKIcospheresFromPoints(Input=tracked); glyphs.Radius = 2.0; glyphs.Subdivisions = 2
gd = Show(glyphs, view); gd.DiffuseColor = [1.0, 0.95, 0.25]
ld = Show(tracked, view); ld.DiffuseColor = [1.0, 1.0, 1.0]; ld.LineWidth = 2.5; ld.Opacity = 0.65
view.CameraPosition = [400, -440, 235]
view.CameraFocalPoint = [128, 128, 8]
view.CameraViewUp = [0, 0, 1]
Render(view)
SaveScreenshot(str(OUT / 'cloud_emd_tracking.png'), view, ImageResolution=[1200, 900])
SaveState(str(OUT / 'cloud_emd_tracking.pvsm'))
print(f'Created {OUT / "cloud_emd_tracking.png"}')
print(f'TTK persistence threshold: {THRESHOLD}; EMD tracks: {len(links)}')
