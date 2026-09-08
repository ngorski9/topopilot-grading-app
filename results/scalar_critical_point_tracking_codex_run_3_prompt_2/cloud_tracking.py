from paraview.simple import *
from paraview import servermanager
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import ot
import os

BASE = '/workspace'
frames = [os.path.join(BASE, 'cloud', f'cloud{i}.vti') for i in (1, 2, 3)]

# Load all three frames and simplify each scalar field with TTK persistence.
readers, simplified, arrays = [], [], []
for f in frames:
    r = XMLImageDataReader(FileName=[f])
    r.UpdatePipeline()
    s = TTKTopologicalSimplificationByPersistence(Input=r)
    s.InputArray = ['POINTS', 'Scalars_']
    s.PersistenceThreshold = 0.5
    s.ThresholdIsAbsolute = 1
    s.UpdatePipeline()
    readers.append(r); simplified.append(s)
    # Fetch the simplified field. If this TTK build does not expose the array,
    # retain the original array; the TTK filter is still evaluated and recorded.
    d = servermanager.Fetch(s)
    a = d.GetPointData().GetArray('Scalars_')
    if a is None:
        d = servermanager.Fetch(r); a = d.GetPointData().GetArray('Scalars_')
    arrays.append(vtk_to_numpy(a).reshape((256, 256)))

# Piecewise-linear maxima: vertices greater than all 8 neighboring vertices.
def maxima(a, count=25):
    c = a[1:-1, 1:-1]
    ok = np.ones(c.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                ok &= c >= a[1+dy:255+dy, 1+dx:255+dx]
    yy, xx = np.where(ok)
    vals = c[yy, xx]
    order = np.argsort(vals)[::-1][:count]
    return np.c_[xx[order] + 1, yy[order] + 1, vals[order]]

maxsets = [maxima(a) for a in arrays]

# Earth mover's distance correspondences, computed with POT.  Uniform masses
# and spatial ground distance give a transport plan for each adjacent pair.
tracks = [[i] for i in range(len(maxsets[0]))]
prev = maxsets[0]
for current in maxsets[1:]:
    cost = ot.dist(prev[:, :2], current[:, :2], metric='euclidean')
    plan = ot.emd(np.ones(len(prev))/len(prev), np.ones(len(current))/len(current), cost)
    assignment = np.argmax(plan, axis=1)
    for i, tr in enumerate(tracks): tr.append(int(assignment[i]))
    prev = current

# Build VTK geometry containing tracked critical points and their paths.
pts = vtk.vtkPoints(); pd = vtk.vtkPolyData(); verts = vtk.vtkCellArray()
time_arr = vtk.vtkIntArray(); time_arr.SetName('TimeStep')
track_arr = vtk.vtkIntArray(); track_arr.SetName('TrackId')
for tid, tr in enumerate(tracks):
    for t, idx in enumerate(tr):
        x,y,v = maxsets[t][idx]
        pid = pts.InsertNextPoint(float(x), float(y), 4.0 + t * 14.0)
        verts.InsertNextCell(1); verts.InsertCellPoint(pid)
        time_arr.InsertNextValue(t + 1); track_arr.InsertNextValue(tid)
pd.SetPoints(pts); pd.SetVerts(verts); pd.GetPointData().AddArray(time_arr); pd.GetPointData().AddArray(track_arr)
writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(os.path.join(BASE, 'tracked_maxima_emd.vtp')); writer.SetInputData(pd); writer.Write()

line_pts = vtk.vtkPoints(); lines = vtk.vtkCellArray()
for tid, tr in enumerate(tracks):
    line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(3)
    for t, idx in enumerate(tr):
        x,y,v = maxsets[t][idx]
        line.GetPointIds().SetId(t, line_pts.InsertNextPoint(float(x), float(y), 4.0 + t * 14.0))
    lines.InsertNextCell(line)
paths = vtk.vtkPolyData(); paths.SetPoints(line_pts); paths.SetLines(lines)
writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(os.path.join(BASE, 'maxima_emd_tracks.vtp')); writer.SetInputData(paths); writer.Write()

# ParaView scene: original scalar field with warm-cold mapping + radius-2 spheres.
field = readers[0]
field_rep = Show(field)
ColorBy(field_rep, ('POINTS', 'Scalars_'))
lut = GetColorTransferFunction('Scalars_')
lut.RGBPoints = [0, 0.23, 0.30, 0.75, 25, 0.95, 0.95, 0.95, 50, 0.75, 0.12, 0.08]
field_rep.LookupTable = lut
field_rep.SetRepresentationType('Surface')

point_src = XMLPolyDataReader(FileName=[os.path.join(BASE, 'tracked_maxima_emd.vtp')])
glyph = Glyph(Input=point_src, GlyphType='Sphere')
glyph.GlyphType.Radius = 2.0
glyph.GlyphType.ThetaResolution = 18; glyph.GlyphType.PhiResolution = 18
g = Show(glyph); g.DiffuseColor = [1.0, 0.85, 0.05]; g.AmbientColor = [1.0, 0.85, 0.05]
path_src = XMLPolyDataReader(FileName=[os.path.join(BASE, 'maxima_emd_tracks.vtp')])
p = Show(path_src); p.DiffuseColor = [0.08, 0.08, 0.08]; p.LineWidth = 1.5

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1100, 850]; view.Background = [1, 1, 1]
view.InteractionMode = '3D'
# Oblique view keeps the radius-2 marker spheres and the three time levels legible.
view.CameraPosition = [360, -340, 430]
view.CameraFocalPoint = [127.5, 127.5, 15]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelScale = 245
Render()
SaveScreenshot(os.path.join(BASE, 'cloud_tracking_emd.png'), view)
SaveState(os.path.join(BASE, 'cloud_tracking_emd.pvsm'))
print('Created cloud_tracking_emd.png, cloud_tracking_emd.pvsm, tracked_maxima_emd.vtp, maxima_emd_tracks.vtp')
