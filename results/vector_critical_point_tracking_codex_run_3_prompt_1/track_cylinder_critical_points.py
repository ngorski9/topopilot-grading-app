"""Track 2-D vector-field critical points in three cylinder VTI frames.

The input field is piecewise linear on the two triangles of every image-data
pixel.  Roots are therefore found exactly in barycentric coordinates.  Partial
optimal transport (POT's partial_wasserstein) links like-index critical points
between consecutive frames, allowing unmatched births/deaths.
"""
from pathlib import Path
import numpy as np
import ot
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from paraview.simple import *

ROOT = Path('/workspace')
DATA = ROOT / 'cylinder'
OUT = ROOT / 'cylinder_tracking'
OUT.mkdir(exist_ok=True)
FRAMES = [DATA / f'cylinder{i}.vti' for i in (1, 2, 3)]

def read_frame(path):
    r = vtk.vtkXMLImageDataReader(); r.SetFileName(str(path)); r.Update()
    im = r.GetOutput()
    dims = im.GetDimensions(); nx, ny = dims[:2]
    u = vtk_to_numpy(im.GetPointData().GetArray('u')).reshape(ny, nx)
    v = vtk_to_numpy(im.GetPointData().GetArray('v')).reshape(ny, nx)
    return im, np.stack((u, v), axis=-1)

def critical_points(im, vec):
    """Return one exact root for every triangle whose affine field contains 0."""
    nx, ny = im.GetDimensions()[:2]
    ox, oy, _ = im.GetOrigin(); sx, sy, _ = im.GetSpacing()
    pts = []
    # split each quad consistently into (00,10,11) and (00,11,01)
    for j in range(ny - 1):
        for i in range(nx - 1):
            for ids in (( (i,j), (i+1,j), (i+1,j+1) ), ((i,j), (i+1,j+1), (i,j+1))):
                f = np.array([vec[y, x] for x, y in ids])
                mat = np.column_stack((f[1] - f[0], f[2] - f[0]))
                try: b = np.linalg.solve(mat, -f[0])
                except np.linalg.LinAlgError: continue
                bary = np.array((1.0 - b.sum(), b[0], b[1]))
                if np.all(bary >= -1e-9) and np.all(bary <= 1 + 1e-9):
                    xy = bary @ np.array([(ox + x*sx, oy + y*sy) for x,y in ids])
                    # affine Jacobian; classify sign(det): saddle (-), extremum (+)
                    xyv = np.array([(x*sx, y*sy) for x,y in ids])
                    J = (f[1:] - f[0]).T @ np.linalg.inv((xyv[1:] - xyv[0]).T)
                    typ = 1 if np.linalg.det(J) > 0 else -1
                    pts.append((xy[0], xy[1], typ))
    # roots on a diagonal/shared edge can be emitted twice: merge near-coincident roots
    merged = []
    for p in pts:
        if not any((p[0]-q[0])**2 + (p[1]-q[1])**2 < 1e-10 for q in merged): merged.append(p)
    return np.array(merged, dtype=float) if merged else np.empty((0,3))

ims, fields = zip(*(read_frame(p) for p in FRAMES))
cps = [critical_points(im, f) for im, f in zip(ims, fields)]
print('critical points per frame:', [len(x) for x in cps])

# Partial OT: spatial cost plus a strong mismatch penalty, transport 90% of the
# mass that both frames can explain.  Keep only substantively transported pairs.
links = []
for a, b in zip(cps[:-1], cps[1:]):
    if not len(a) or not len(b): links.append([]); continue
    cost = ot.dist(a[:, :2], b[:, :2], metric='sqeuclidean')
    cost += (a[:,2,None] != b[None,:,2]) * 2.5e5
    wa, wb = np.full(len(a), 1/len(a)), np.full(len(b), 1/len(b))
    gamma = ot.partial.partial_wasserstein(wa, wb, cost, m=0.90)
    ij = np.argwhere(gamma > 0.20 * gamma.max())
    links.append([(int(i), int(j)) for i,j in ij])
print('partial-OT links:', [len(x) for x in links])

def point_polydata():
    points, verts, time, kind, pid = vtk.vtkPoints(), vtk.vtkCellArray(), [], [], []
    for t, arr in enumerate(cps):
        for k, (x,y,c) in enumerate(arr):
            p = points.InsertNextPoint(x,y,0); verts.InsertNextCell(1); verts.InsertCellPoint(p)
            time.append(t); kind.append(c); pid.append(k)
    pd = vtk.vtkPolyData(); pd.SetPoints(points); pd.SetVerts(verts)
    for name, values in [('TimeStep',time),('CriticalType',kind),('LocalId',pid)]:
        ar = numpy_to_vtk(np.asarray(values)); ar.SetName(name); pd.GetPointData().AddArray(ar)
    return pd

def line_polydata():
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray(); segtime = []
    for t, pairs in enumerate(links):
        for i,j in pairs:
            ids = vtk.vtkIdList(); ids.SetNumberOfIds(2)
            ids.SetId(0, points.InsertNextPoint(cps[t][i,0], cps[t][i,1], 1.0))
            ids.SetId(1, points.InsertNextPoint(cps[t+1][j,0], cps[t+1][j,1], 1.0))
            lines.InsertNextCell(ids); segtime.append(t)
    pd = vtk.vtkPolyData(); pd.SetPoints(points); pd.SetLines(lines)
    ar = numpy_to_vtk(np.asarray(segtime)); ar.SetName('FromTimeStep'); pd.GetCellData().AddArray(ar)
    return pd

def write(pd, name):
    w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(OUT/name)); w.SetInputData(pd); w.Write()
write(point_polydata(), 'tracked_critical_points.vtp')
write(line_polydata(), 'partial_ot_tracks.vtp')

# Store a 2-component vector copy of frame 1 for straightforward visualisation.
base = vtk.vtkImageData(); base.DeepCopy(ims[0])
vf = numpy_to_vtk(np.ascontiguousarray(fields[0].reshape(-1,2)), deep=True); vf.SetName('Velocity'); vf.SetNumberOfComponents(2)
base.GetPointData().AddArray(vf)
w=vtk.vtkXMLImageDataWriter(); w.SetFileName(str(OUT/'cylinder_vector_field_t0.vti')); w.SetInputData(base); w.Write()

# ParaView presentation: sampled arrows are the original t=0 field, while
# colored spheres/lines show the three-frame critical-point tracking result.
field = XMLImageDataReader(registrationName='Cylinder vector field (t=0)', FileName=[str(OUT/'cylinder_vector_field_t0.vti')])
field.PointArrayStatus = ['Velocity']
sample = MaskPoints(registrationName='Vector field sample', Input=field); sample.OnRatio=120
glyph = Glyph(registrationName='Original vector field', Input=sample, GlyphType='Arrow')
glyph.OrientationArray=['POINTS','Velocity']; glyph.ScaleArray=['POINTS','Velocity']; glyph.ScaleFactor=40
tracks = XMLPolyDataReader(registrationName='Partial optimal transport tracks', FileName=[str(OUT/'partial_ot_tracks.vtp')])
points = XMLPolyDataReader(registrationName='Tracked critical points', FileName=[str(OUT/'tracked_critical_points.vtp')])
view=GetActiveViewOrCreate('RenderView'); view.ViewSize=[1200,800]; view.Background=[0.08,0.09,0.12]
gd=Show(glyph,view); gd.DiffuseColor=[0.72,0.75,0.78]; gd.LineWidth=1.5
td=Show(tracks,view); td.DiffuseColor=[1.0,0.75,0.15]; td.LineWidth=5.0
pd=Show(points,view); pd.Representation='Points'; pd.PointSize=22; ColorBy(pd, ('POINTS','TimeStep'))
lut=GetColorTransferFunction('TimeStep'); lut.RGBPoints=[0,0.25,0.7,1,1,0.2,0.9,0.3,2,1,0.25,0.25]
pd.SetScalarBarVisibility(view,True)
view.CameraPosition=[75,225,500]; view.CameraFocalPoint=[75,225,0]; view.CameraViewUp=[0,1,0]
Render(); SaveScreenshot(str(OUT/'cylinder_partial_ot_tracking.png'), view)
SaveState(str(OUT/'cylinder_partial_ot_tracking.pvsm'))
print('wrote', OUT)
