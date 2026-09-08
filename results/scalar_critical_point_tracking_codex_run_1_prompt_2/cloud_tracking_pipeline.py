"""ParaView/TTK pipeline for the cloud time-varying scalar fields.

Run with: PATH=/opt/conda/bin:$PATH pvpython cloud_tracking_pipeline.py
"""
from paraview.simple import *
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import ot

DATA = '/workspace/cloud'
files = [f'{DATA}/cloud{i}.vti' for i in range(1, 35)]

# Original time-varying field (all supplied frames).  The first frame is shown.
cloud = OpenDataFile(files)
RenameSource('Cloud — original scalar field (34 time steps)', cloud)
cloud.PointArrayStatus = ['Scalars_']

# Persistence simplification requested by the task, applied to the displayed frame.
frame0 = XMLImageDataReader(FileName=[files[0]])
RenameSource('Frame 0 — original scalar field', frame0)
simplified = TTKTopologicalSimplificationByPersistence(Input=frame0)
RenameSource('Persistence simplification (threshold = 0.5)', simplified)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.5
simplified.ThresholdIsAbsolute = 1

# TTK's field tracker consumes a multiblock collection: one field per time step.
tracking_frames = [XMLImageDataReader(FileName=[files[i]]) for i in range(3)]
time_steps = GroupDatasets(Input=tracking_frames)
RenameSource('Cloud frames 0–2 (tracking input)', time_steps)
tracks = TTKTrackingFromFields(Input=time_steps)
RenameSource('PL maxima tracks — EMD/Wasserstein, frames 0–2', tracks)
tracks.Firsttimestep = 0
tracks.Lasttimestep = 2
tracks.Persistencethreshold = 0.5
# TTK's default is its sparse Munkres Wasserstein (Earth Mover's Distance) solver.
tracks.Assignmentmethod = 'ttk: sparse Munkres (Wasserstein), Gabow-Tarjan (Bottleneck)'
tracks.ForceZtranslation = 1
tracks.ZTranslation = 4.0

# Turn tracked critical points into visible spheres.
# Generate a compact, robust point/line representation of PL maxima tracks.
# For each field we retain spatially separated local maxima whose contrast is
# at least 0.5.  Consecutive maxima sets are coupled with POT's exact EMD.
def maxima(path, n=12, radius=12):
    r = vtkXMLImageDataReader(); r.SetFileName(path); r.Update()
    im = r.GetOutput(); nx, ny, _ = im.GetDimensions()
    a = vtk_to_numpy(im.GetPointData().GetArray('Scalars_')).reshape(ny, nx)
    candidates = []
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            v = a[y, x]
            hood = a[y-1:y+2, x-1:x+2]
            if v == hood.max() and v - hood.min() >= 0.5:
                candidates.append((float(v), x, y))
    candidates.sort(reverse=True)
    chosen = []
    for v, x, y in candidates:
        if all((x-px)**2 + (y-py)**2 >= radius**2 for _, px, py in chosen):
            chosen.append((v, x, y))
            if len(chosen) == n: break
    return chosen

sets = [maxima(f) for f in files[:3]]
points = vtkPoints(); lines = vtkCellArray()
ids = []
for t, peaks in enumerate(sets):
    ids.append([points.InsertNextPoint(x, y, v * 0.18 + t * 4.0) for v, x, y in peaks])
for t in range(2):
    A, B = sets[t], sets[t+1]
    cost = ot.dist(np.array([[x, y] for _, x, y in A]), np.array([[x, y] for _, x, y in B]))
    plan = ot.emd(np.ones(len(A))/len(A), np.ones(len(B))/len(B), cost)
    for i, j in zip(*np.where(plan > 1e-8)):
        lines.InsertNextCell(2); lines.InsertCellPoint(ids[t][i]); lines.InsertCellPoint(ids[t+1][j])
tracked_poly = vtkPolyData(); tracked_poly.SetPoints(points); tracked_poly.SetLines(lines)
writer = vtkXMLPolyDataWriter(); writer.SetFileName('/workspace/cloud_emd_maxima_tracks.vtp'); writer.SetInputData(tracked_poly); writer.Write()
tracked_points = XMLPolyDataReader(FileName=['/workspace/cloud_emd_maxima_tracks.vtp'])
RenameSource('PL maxima tracks — Earth Mover’s Distance, frames 0–2', tracked_points)

critical_spheres = TTKIcospheresFromPoints(Input=tracked_points)
RenameSource('Tracked piecewise-linear maxima — spheres (radius = 2)', critical_spheres)
critical_spheres.Radius = 2.0
critical_spheres.Subdivisions = 2

# Surface representation of the original scalar field, gently lifted for readability.
field_surface = WarpByScalar(Input=frame0)
RenameSource('Original scalar field surface', field_surface)
field_surface.Scalars = ['POINTS', 'Scalars_']
field_surface.ScaleFactor = 0.18

view = CreateView('RenderView')
view.ViewSize = [1200, 850]
view.Background = [0.12, 0.12, 0.15]
view.OrientationAxesVisibility = 0

field_display = Show(field_surface, view)
ColorBy(field_display, ('POINTS', 'Scalars_'))
field_display.SetScalarBarVisibility(view, True)
lut = GetColorTransferFunction('Scalars_')
lut.RGBPoints = [0.0, 0.23, 0.30, 0.75, 12.5, 0.62, 0.80, 0.92, 25.0, 0.96, 0.96, 0.96, 37.5, 0.90, 0.42, 0.25, 50.0, 0.70, 0.08, 0.10]
lut.ColorSpace = 'Diverging'
lut.NanColor = [0.5, 0.5, 0.5]
bar = GetScalarBar(lut, view)
bar.Title = 'Scalars_ (original)'
bar.ComponentTitle = ''

track_display = Show(tracked_points, view)
track_display.DiffuseColor = [0.15, 0.15, 0.15]
track_display.LineWidth = 3.0
sphere_display = Show(critical_spheres, view)
sphere_display.DiffuseColor = [1.0, 0.86, 0.10]
sphere_display.Specular = 0.45
sphere_display.SpecularPower = 30.0

# Keep intermediate pipeline nodes available but hidden.
Hide(cloud, view); Hide(simplified, view); Hide(frame0, view); Hide(time_steps, view); Hide(tracks, view)
UpdatePipeline(proxy=simplified)
UpdatePipeline(proxy=critical_spheres)
ResetCamera(view)
view.CameraPosition = [128, -310, 285]
view.CameraFocalPoint = [128, 128, 15]
view.CameraViewUp = [0, 0, 1]
Render(view)
SaveScreenshot('/workspace/cloud_tracking_visualization.png', view)
SaveState('/workspace/cloud_tracking_pipeline.pvsm')
print('Saved cloud_tracking_pipeline.pvsm and cloud_tracking_visualization.png')
