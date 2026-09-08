from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
import numpy as np
import ot

BASE = '/workspace'
FILES = [f'{BASE}/cloud/cloud{i}.vti' for i in (1, 2, 3)]
OUT = f'{BASE}/output'

def simplified_maxima(filename):
    reader = XMLImageDataReader(FileName=[filename])
    simp = TTKTopologicalSimplificationByPersistence(Input=reader)
    simp.InputArray = ['POINTS', 'Scalars_']
    simp.PersistenceThreshold = 0.5
    simp.ThresholdIsAbsolute = 1
    simp.ThreadNumber = 4
    simp.UpdatePipeline()
    image = servermanager.Fetch(simp)
    values = vtk_to_numpy(image.GetPointData().GetArray('Scalars_')).reshape((256, 256))
    # Maxima of the persistence-simplified PL field (8-connected neighborhood).
    pad = np.pad(values, 1, mode='edge')
    ismax = np.ones_like(values, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                ismax &= values >= pad[1+dy:257+dy, 1+dx:257+dx]
    # retain one representative from plateaus and keep a compact, meaningful set
    candidates = np.argwhere(ismax & (values >= 1))
    order = np.argsort(values[candidates[:,0], candidates[:,1]])[::-1]
    picked = []
    for k in order:
        y, x = candidates[k]
        if all((x-px)**2 + (y-py)**2 > 25 for px, py, _ in picked):
            picked.append((int(x), int(y), float(values[y, x])))
        if len(picked) == 30:
            break
    return reader, simp, picked

readers, simplifications, maxima = zip(*[simplified_maxima(f) for f in FILES])

# EMD transport plan between consecutive maxima.  The maximum-mass edge from
# every source critical point defines its piecewise-linear temporal successor.
tracks = []
for step in range(2):
    a, b = maxima[step], maxima[step+1]
    A = np.array([[p[0], p[1], p[2]] for p in a], float)
    B = np.array([[p[0], p[1], p[2]] for p in b], float)
    cost = ot.dist(A, B, metric='euclidean')
    plan = ot.emd(np.ones(len(a))/len(a), np.ones(len(b))/len(b), cost)
    tracks.append(np.argmax(plan, axis=1))

# Points and PL track segments.  Z separates time steps while retaining the
# original x/y domain, making all three time samples visible with the field.
points = vtk.vtkPoints(); verts = vtk.vtkCellArray(); lines = vtk.vtkCellArray()
time_array = vtk.vtkIntArray(); time_array.SetName('TimeStep')
scalar_array = vtk.vtkDoubleArray(); scalar_array.SetName('CriticalValue')
ids = []
for t, crits in enumerate(maxima):
    row = []
    for x, y, val in crits:
        pid = points.InsertNextPoint(x, y, t * 8.0 + 2.0)
        row.append(pid); verts.InsertNextCell(1); verts.InsertCellPoint(pid)
        time_array.InsertNextValue(t); scalar_array.InsertNextValue(val)
    ids.append(row)
for t, mapping in enumerate(tracks):
    for i, j in enumerate(mapping):
        line = vtk.vtkLine(); line.GetPointIds().SetId(0, ids[t][i]); line.GetPointIds().SetId(1, ids[t+1][j])
        lines.InsertNextCell(line)
poly = vtk.vtkPolyData(); poly.SetPoints(points); poly.SetVerts(verts); poly.SetLines(lines)
poly.GetPointData().AddArray(time_array); poly.GetPointData().AddArray(scalar_array)
writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(f'{OUT}/emd_maxima_tracks.vtp'); writer.SetInputData(poly); writer.Write()

# Persist the exact TTK processing/visualization pipeline in a ParaView state.
track_source = XMLPolyDataReader(FileName=[f'{OUT}/emd_maxima_tracks.vtp'])
# Show the original scalar field; the hidden TTK filters above provide the
# persistence-simplified fields used to extract and track the maxima.
field = readers[0]
view = CreateView('RenderView'); view.ViewSize = [1200, 850]; view.Background = [0.08, 0.08, 0.10]
field_display = Show(field, view); ColorBy(field_display, ('POINTS', 'Scalars_'))
field_display.Representation = 'Surface'; field_display.Opacity = 0.88
lut = GetColorTransferFunction('Scalars_'); lut.RGBPoints = [0.0, 0.23,0.30,0.75, 25.0, 0.97,0.97,0.97, 50.0, 0.70,0.12,0.10]
lut.ColorSpace = 'Lab'; field_display.LookupTable = lut
track_display = Show(track_source, view); track_display.Representation = 'Surface'; track_display.DiffuseColor = [1.0, 0.85, 0.2]; track_display.LineWidth = 2.5
spheres = Glyph(Input=track_source, GlyphType='Sphere'); spheres.GlyphType.Radius = 2.0; spheres.GlyphType.ThetaResolution = 20; spheres.GlyphType.PhiResolution = 20
spheres.ScaleArray = ['POINTS', '']; spheres.ScaleFactor = 1.0; spheres.GlyphMode = 'All Points'
sphere_display = Show(spheres, view); ColorBy(sphere_display, ('POINTS', 'TimeStep'))
tlut = GetColorTransferFunction('TimeStep'); tlut.RGBPoints = [0, 0.2,0.4,0.8, 1, 1.0,0.95,0.3, 2, 0.8,0.15,0.1]
sphere_display.LookupTable = tlut
view.CameraPosition = [128, -330, 310]; view.CameraFocalPoint = [128, 128, 8]; view.CameraViewUp = [0, 0, 1]
Render(view); SaveScreenshot(f'{OUT}/cloud_emd_tracking.png', view, ImageResolution=[1200,850]); SaveState(f'{OUT}/cloud_emd_tracking.pvsm')
with open(f'{OUT}/summary.txt', 'w') as f:
    f.write('TTK persistence simplification threshold: 0.5 (absolute)\n')
    f.write('EMD / optimal transport maxima tracking across cloud1, cloud2, cloud3\n')
    f.write('Sphere radius: 2\n')
    f.write('Maxima per step: ' + ', '.join(str(len(x)) for x in maxima) + '\n')
