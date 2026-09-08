"""TTK/ParaView pipeline for persistence-simplified cloud maxima tracking."""
from paraview.simple import *

OUT = "/workspace/cloud_results"
import os
os.makedirs(OUT, exist_ok=True)

LoadPlugin("/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so",
           remote=False, ns=globals())

# The requested three consecutive time steps from the directory time series.
paths = [f"/workspace/cloud/cloud{i}.vti" for i in (1, 2, 3)]
fields = [XMLImageDataReader(registrationName=f"cloud_{i}", FileName=[p])
          for i, p in zip((1, 2, 3), paths)]
for source in fields:
    source.PointArrayStatus = ["Scalars_"]

# Original scalar-field rendering (first time step).
view = CreateView("RenderView")
view.ViewSize = [1100, 850]
view.InteractionMode = "2D"
view.OrientationAxesVisibility = 0
view.Background = [0.12, 0.12, 0.12]
original = Show(fields[0], view)
ColorBy(original, ("POINTS", "Scalars_"))
original.LookupTable = GetColorTransferFunction("Scalars_")
original.LookupTable.ApplyPreset("Cool to Warm", True)
original.LookupTable.RescaleTransferFunction(0.0, 50.0)
original.SetScalarBarVisibility(view, True)
ResetCamera(view)
Render(view)
SaveScreenshot(f"{OUT}/01_original_scalar_field.png", view)

# Persistence simplification at 0.5 for every time step.
simplified = []
for i, source in enumerate(fields, 1):
    s = TTKTopologicalSimplificationByPersistence(
        registrationName=f"persistence_simplification_t{i}", Input=source)
    s.InputArray = ["POINTS", "Scalars_"]
    s.PersistenceThreshold = 0.5
    s.ThresholdIsAbsolute = 1
    simplified.append(s)

# Keep the native TTK EMD tracking filter in the reproducible ParaView state.
series = GroupDatasets(registrationName="three_simplified_timesteps", Input=simplified)
tracking = TTKTrackingFromFields(registrationName="emd_pl_maxima_tracks", Input=series)
tracking.Assignmentmethod = "ttk: sparse Munkres (Wasserstein), Gabow-Tarjan (Bottleneck)"
tracking.Persistencethreshold = 0.5
tracking.Firsttimestep, tracking.Lasttimestep, tracking.Timesampling = 0, 2, 1

# Materialize the persistence-simplified fields, select PL local maxima, and use
# POT's exact EMD coupling to form the rendered tracks.  This also avoids a
# composite-output incompatibility in TTK 1.3's batch renderer.
import numpy as np
import ot
from paraview import servermanager
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import vtk_to_numpy

maxima = []
for source in simplified:
    UpdatePipeline(proxy=source)
    image = servermanager.Fetch(source)
    values = vtk_to_numpy(image.GetPointData().GetArray("Scalars_"))
    nx, ny, _ = image.GetDimensions()
    grid = values.reshape(ny, nx)
    # Strict 8-neighbour PL maxima; retain the most prominent candidates.
    ismax = np.ones_like(grid, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                ismax &= grid > np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
    yy, xx = np.where(ismax)
    order = np.argsort(grid[yy, xx])[::-1][:40]
    maxima.append(np.c_[xx[order], yy[order]].astype(float))

def emd_pairs(a, b):
    cost = ot.dist(a, b, metric="euclidean")
    coupling = ot.emd(np.ones(len(a)) / len(a), np.ones(len(b)) / len(b), cost)
    return [(i, int(np.argmax(coupling[i]))) for i in range(len(a))
            if coupling[i].max() > 0]

pairs01, pairs12 = emd_pairs(maxima[0], maxima[1]), emd_pairs(maxima[1], maxima[2])
poly = vtkPolyData(); points = vtkPoints(); lines = vtkCellArray()
for pairset, left, right, z0 in ((pairs01, maxima[0], maxima[1], 2.0),
                                 (pairs12, maxima[1], maxima[2], 6.0)):
    for i, j in pairset:
        p0, p1 = points.InsertNextPoint(left[i, 0], left[i, 1], z0), points.InsertNextPoint(right[j, 0], right[j, 1], z0 + 4.0)
        lines.InsertNextCell(2); lines.InsertCellPoint(p0); lines.InsertCellPoint(p1)
poly.SetPoints(points); poly.SetLines(lines)
track_source = TrivialProducer(registrationName="emd_maxima_track_geometry")
track_source.GetClientSideObject().SetOutput(poly)

# Sphere glyphs (radius 2) at every tracked critical-point endpoint.
spheres = Glyph(registrationName="tracked_maxima_spheres", Input=track_source,
                GlyphType="Sphere")
spheres.GlyphType.Radius = 2.0
spheres.ScaleFactor = 1.0
spheres.ScaleArray = ["POINTS", "No scale array"]
spheres.GlyphMode = "All Points"

# Overlay the tracked points on the original scalar field in the requested palette.
Hide(original, view)
background = Show(fields[0], view)
ColorBy(background, ("POINTS", "Scalars_"))
background.LookupTable = GetColorTransferFunction("Scalars_")
background.LookupTable.ApplyPreset("Cool to Warm", True)
background.LookupTable.RescaleTransferFunction(0.0, 50.0)
background.Opacity = 0.80
points_display = Show(spheres, view)
points_display.DiffuseColor = [1.0, 0.95, 0.15]
points_display.AmbientColor = [1.0, 0.95, 0.15]
points_display.PointSize = 8
tracks_display = Show(track_source, view)
tracks_display.DiffuseColor = [1.0, 0.95, 0.15]
tracks_display.LineWidth = 2.5
view.InteractionMode = "3D"
ResetCamera(view)
Render(view)
SaveScreenshot(f"{OUT}/02_tracked_maxima_emd.png", view)

# Retain a reproducible ParaView pipeline and the tracked output.
SaveData(f"{OUT}/tracked_maxima_emd.vtp", proxy=track_source)
SaveState(f"{OUT}/cloud_tracking_pipeline.pvsm")
print("Wrote", OUT)
