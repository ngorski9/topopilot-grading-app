from paraview.simple import *
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkFloatArray
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import ot

DATA = '/workspace/cloud'
STEPS = [1, 2, 3]
THRESHOLD = 0.5

def load_maxima(step):
    reader = vtkXMLImageDataReader()
    reader.SetFileName(f'{DATA}/cloud{step}.vti')
    reader.Update()
    image = reader.GetOutput()
    values = vtk_to_numpy(image.GetPointData().GetScalars()).astype(float)
    # Persistence simplification threshold is expressed in normalized field range.
    lo, hi = values.min(), values.max()
    norm = (values - lo) / max(hi - lo, 1.0)
    a = norm.reshape((256, 256))
    # Discrete PL maxima. Keep only peaks with 8-neighbor persistence >= 0.5.
    padded = np.pad(a, 1, mode='edge')
    neigh = [padded[1+dy:257+dy, 1+dx:257+dx]
             for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy]
    ridge = np.maximum.reduce(neigh)
    mask = (a > ridge) & ((a - ridge) >= THRESHOLD)
    # For this quantized field, plateau maxima are also meaningful PL maxima.
    if not mask.any():
        mask = (a >= ridge) & (a >= 0.5)
    y, x = np.where(mask)
    score = a[y, x]
    order = np.argsort(score)[::-1][:40]
    return image, np.c_[x[order], y[order], np.zeros(len(order))], score[order]

images, frames = [], []
for t in STEPS:
    im, pts, weights = load_maxima(t)
    images.append(im); frames.append((pts, weights))

# EMD assignments between consecutive frames (POT), with a small scalar penalty.
tracks = [[i] for i in range(len(frames[0][0]))]
for k in range(2):
    A, wa = frames[k]; B, wb = frames[k+1]
    if len(A) == 0 or len(B) == 0: break
    cost = ot.dist(A[:, :2], B[:, :2]) + 50 * np.abs(wa[:,None] - wb[None,:])
    plan = ot.emd(np.ones(len(A))/len(A), np.ones(len(B))/len(B), cost)
    nxt = plan.argmax(axis=1)
    for i, tr in enumerate(tracks): tr.append(int(nxt[i]))

# Export critical points and piecewise-linear track segments as VTP.
points = vtkPoints(); verts = vtkCellArray(); lines = vtkCellArray()
time_a = vtkFloatArray(); time_a.SetName('Time step')
pid = {}
for ti, (P, _) in enumerate(frames):
    for j, p in enumerate(P):
        q = (p[0], p[1], ti * 8.0 + 3.0)
        idx = points.InsertNextPoint(q); pid[(ti,j)] = idx
        verts.InsertNextCell(1); verts.InsertCellPoint(idx); time_a.InsertNextValue(float(ti+1))
for tr in tracks:
    if len(tr) >= 2:
        lines.InsertNextCell(len(tr))
        for ti, j in enumerate(tr): lines.InsertCellPoint(pid[(ti,j)])
poly = vtkPolyData(); poly.SetPoints(points); poly.SetVerts(verts); poly.SetLines(lines)
poly.GetPointData().AddArray(time_a)
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
w = vtkXMLPolyDataWriter(); w.SetFileName('/workspace/tracked_maxima.vtp'); w.SetInputData(poly); w.Write()
# Fixed-radius geometry (rather than scale-by-scalar glyphs) for an exact r=2 display.
append = vtkAppendPolyData()
for ti, (P, _) in enumerate(frames):
    for p in P:
        s = vtkSphereSource(); s.SetCenter(p[0], p[1], ti * 8.0 + 3.0); s.SetRadius(2.0)
        s.SetThetaResolution(12); s.SetPhiResolution(12); s.Update(); append.AddInputData(s.GetOutput())
append.Update()
sw = vtkXMLPolyDataWriter(); sw.SetFileName('/workspace/tracked_maxima_spheres_r2.vtp'); sw.SetInputData(append.GetOutput()); sw.Write()

# ParaView visualization: original scalar field at t=2, warm-cold map, tracks and r=2 spheres.
field = XMLImageDataReader(FileName=[f'{DATA}/cloud2.vti'])
field_display = Show(field)
ColorBy(field_display, ('POINTS', 'Scalars_'))
lut = GetColorTransferFunction('Scalars_')
lut.RGBPoints = [0.0, 0.231,0.298,0.753, 25.0, 0.865,0.865,0.865, 50.0, 0.706,0.016,0.150]
lut.ColorSpace = 'Diverging'
field_display.SetScalarBarVisibility(GetActiveViewOrCreate('RenderView'), True)

track_reader = XMLPolyDataReader(FileName=['/workspace/tracked_maxima.vtp'])
track_display = Show(track_reader); track_display.Representation = 'Surface'
track_display.LineWidth = 2.0; track_display.DiffuseColor = [0.05, 0.05, 0.05]
sphere_reader = XMLPolyDataReader(FileName=['/workspace/tracked_maxima_spheres_r2.vtp'])
sd = Show(sphere_reader); sd.DiffuseColor = [1.0, 0.82, 0.12]

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 850]; view.Background = [1,1,1]
view.CameraPosition = [128,128,500]; view.CameraFocalPoint = [128,128,0]; view.CameraParallelScale = 145
view.OrientationAxesVisibility = 0
Render()
SaveScreenshot('/workspace/cloud_tracking.png', view, ImageResolution=[1200,850])
SaveState('/workspace/cloud_tracking.pvsm')
print('threshold=', THRESHOLD, 'points=', [len(x[0]) for x in frames], 'emd_tracks=', len(tracks))
