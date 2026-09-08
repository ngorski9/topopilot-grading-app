"""
TTK + Python-Optimal-Transport pipeline on the ./cloud time-varying scalar field:

1. Import the cloud time series (.vti, 34 timesteps of a 256x256 2D field).
2. Show the original scalar field.
3. Persistence-simplify the maxima (threshold = 0.5) with
   TTKTopologicalSimplificationByPersistence.
4. Extract the piecewise-linear (PL) maxima of the 3 first timesteps with
   TTKScalarFieldCriticalPoints, then track them across the 3 timesteps by
   solving an Earth Mover's Distance (Wasserstein) transport problem between
   consecutive frames with the Python Optimal Transport (POT) library.
   (TTK's own TTKTrackingFromFields filter segfaults on composite/multiblock
   input in this environment - see /tmp/repro.py / gdb backtrace - so the
   EMD matching step is performed directly with POT instead.)
5. Visualize the tracked maxima (sphere glyphs, radius=2) together with the
   scalar field, using the "Warm to Cold" colormap.
"""
import glob
import os
import re

import numpy as np
import ot
from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy

paraview.simple._DisableFirstRenderCameraReset()

OUT = "/workspace/out"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Import the cloud time-varying dataset
# ---------------------------------------------------------------------------
files = sorted(
    glob.glob("/workspace/cloud/cloud*.vti"),
    key=lambda p: int(re.search(r"cloud(\d+)\.vti", p).group(1)),
)
print("Found %d timesteps" % len(files))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ["Scalars_"]
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1000, 1000]

# ---------------------------------------------------------------------------
# 2. Show the original scalar field (first timestep)
# ---------------------------------------------------------------------------
scene.AnimationTime = reader.TimestepValues[0]
origDisplay = Show(reader, view)
ColorBy(origDisplay, ("POINTS", "Scalars_"))
origDisplay.RescaleTransferFunctionToDataRange(True)
origLUT = GetColorTransferFunction("Scalars_")
origLUT.ApplyPreset("Warm to Cold", True)
origDisplay.SetScalarBarVisibility(view, True)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT, "01_original_scalar_field.png"), view)
print("Saved original scalar field screenshot")

# ---------------------------------------------------------------------------
# 3+4. Persistence simplification (threshold=0.5) + maxima extraction,
#      for the first 3 timesteps
# ---------------------------------------------------------------------------
N_TRACK_STEPS = 3
maxima_per_step = []  # list of (Nx3 float array of xyz, values array)
simplify_last = None

for i in range(N_TRACK_STEPS):
    r = XMLImageDataReader(FileName=[files[i]])
    r.PointArrayStatus = ["Scalars_"]

    s = TTKTopologicalSimplificationByPersistence(Input=r)
    s.InputArray = ["POINTS", "Scalars_"]
    s.PairType = "Maximum-Saddle"
    s.PersistenceThreshold = 0.5
    s.ThresholdIsAbsolute = 0
    simplify_last = s

    cp = TTKScalarFieldCriticalPoints(Input=s)
    cp.ScalarField = ["POINTS", "Scalars_"]
    cp.UpdatePipeline()

    polydata = servermanager.Fetch(cp)
    pts = vtk_to_numpy(polydata.GetPoints().GetData())
    ctype = vtk_to_numpy(polydata.GetPointData().GetArray("CriticalType"))
    vals = vtk_to_numpy(polydata.GetPointData().GetArray("Scalars_"))

    is_max = ctype == 3  # TTK CriticalType encoding: 0=min, 1=saddle, 3=max, 4=multi-saddle
    maxima_pts = pts[is_max]
    maxima_vals = vals[is_max]
    maxima_per_step.append((maxima_pts, maxima_vals))
    print("timestep %d: %d PL maxima (after persistence simplification)" % (i, len(maxima_pts)))

    if i == 0:
        # show the persistence-simplified scalar field (threshold=0.5) for timestep 0
        simpDisplay = Show(s, view)
        Hide(reader, view)
        ColorBy(simpDisplay, ("POINTS", "Scalars_"))
        simpDisplay.RescaleTransferFunctionToDataRange(True)
        simpLUT = GetColorTransferFunction("Scalars_")
        simpLUT.ApplyPreset("Warm to Cold", True)
        simpDisplay.SetScalarBarVisibility(view, True)
        Render()
        SaveScreenshot(os.path.join(OUT, "02_persistence_simplified.png"), view)
        Hide(s, view)
        print("Saved persistence-simplified scalar field screenshot")

# ---------------------------------------------------------------------------
# Track the maxima across the 3 timesteps using Earth Mover's Distance (POT)
# ---------------------------------------------------------------------------
def emd_match(pts_a, pts_b):
    """Solve the EMD/Wasserstein transport plan between two maxima point sets
    (uniform mass, Euclidean ground cost) and return, for each point in A,
    the index of the point in B receiving the largest transported mass."""
    na, nb = len(pts_a), len(pts_b)
    a = np.ones(na) / na
    b = np.ones(nb) / nb
    M = ot.dist(pts_a, pts_b, metric="euclidean")
    plan = ot.emd(a, b, M)
    return np.argmax(plan, axis=1)

track_points = []   # (x, y, z, timestep, value, track_id)
next_track_id = 0
active_tracks = {}  # index within maxima_per_step[0] -> track_id

pts0, vals0 = maxima_per_step[0]
for idx in range(len(pts0)):
    active_tracks[idx] = next_track_id
    x, y, z = pts0[idx]
    track_points.append((x, y, z, 0, vals0[idx], next_track_id))
    next_track_id += 1

for t in range(N_TRACK_STEPS - 1):
    pts_a, vals_a = maxima_per_step[t]
    pts_b, vals_b = maxima_per_step[t + 1]
    match = emd_match(pts_a, pts_b)  # index in B for each point in A

    new_active = {}
    for a_idx, tid in active_tracks.items():
        b_idx = match[a_idx]
        new_active[b_idx] = tid  # last writer wins if several A's map to same B
    for b_idx, tid in new_active.items():
        x, y, z = pts_b[b_idx]
        track_points.append((x, y, z, t + 1, vals_b[b_idx], tid))
    active_tracks = new_active
    print("matched %d maxima at t=%d -> t=%d via EMD" % (len(active_tracks), t, t + 1))

track_points = np.array(track_points)
print("Total tracked maxima instances across %d timesteps: %d" % (N_TRACK_STEPS, len(track_points)))

# ---------------------------------------------------------------------------
# 5. Visualize tracked maxima (spheres, radius=2) together with the scalar field
# ---------------------------------------------------------------------------
import vtk

trackedPD = vtk.vtkPolyData()
vpoints = vtk.vtkPoints()
scalarArr = vtk.vtkFloatArray()
scalarArr.SetName("Scalars_")
timeArr = vtk.vtkIntArray()
timeArr.SetName("Timestep")
trackArr = vtk.vtkIntArray()
trackArr.SetName("TrackId")

for x, y, z, t, val, tid in track_points:
    vpoints.InsertNextPoint(float(x), float(y), float(z))
    scalarArr.InsertNextValue(float(val))
    timeArr.InsertNextValue(int(t))
    trackArr.InsertNextValue(int(tid))

trackedPD.SetPoints(vpoints)
trackedPD.GetPointData().AddArray(scalarArr)
trackedPD.GetPointData().AddArray(timeArr)
trackedPD.GetPointData().AddArray(trackArr)
trackedPD.GetPointData().SetActiveScalars("Scalars_")

trackedSource = TrivialProducer(registrationName="TrackedMaxima")
trackedSource.GetClientSideObject().SetOutput(trackedPD)
trackedSource.UpdatePipeline()

Hide(reader, view)
fieldDisplay = Show(reader, view)
ColorBy(fieldDisplay, ("POINTS", "Scalars_"))
fieldDisplay.RescaleTransferFunctionToDataRange(True)
fieldLUT = GetColorTransferFunction("Scalars_")
fieldLUT.ApplyPreset("Warm to Cold", True)
fieldDisplay.SetScalarBarVisibility(view, True)

glyph = Glyph(Input=trackedSource, GlyphType="Sphere")
glyph.GlyphType.Radius = 2.0
glyph.ScaleArray = ["POINTS", "No scale array"]
glyph.GlyphMode = "All Points"

glyphDisplay = Show(glyph, view)
ColorBy(glyphDisplay, ("POINTS", "Scalars_"))
glyphLUT = GetColorTransferFunction("Scalars_")
glyphLUT.ApplyPreset("Warm to Cold", True)
glyphDisplay.SetScalarBarVisibility(view, True)

view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT, "03_tracked_maxima_with_field.png"), view)
print("Saved tracked-maxima + scalar field screenshot")

print("Done. Outputs in", OUT)
