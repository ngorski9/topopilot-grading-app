#!/usr/bin/env pvpython
"""
TTK pipeline on the time-dependent 'cloud' dataset:
1. Load the time series (34 .vti slices, array 'Scalars_').
2. Render the original scalar field.
3. Persistence simplification (threshold = 0.5) on the maxima-saddle pairs.
4. Extract persistent maxima for the first 3 time steps and track them
   using an Earth Mover's Distance (optimal transport) assignment between
   consecutive time steps (Python Optimal Transport / POT), building
   piecewise-linear tracks.
5. Visualize the tracked critical points (sphere radius = 2) together with
   the scalar field, using a warm-cold colormap.

Note: TTKTrackingFromPersistenceDiagrams segfaults in this environment's
TTK 1.3.0 build regardless of whether it is driven through the ParaView
proxy layer or the raw VTK/ttk Python bindings (verified with minimal
repros), so the EMD-based tracking of maxima is performed directly with
the "Python Optimal Transport" (POT) library instead.
"""
import glob
import os
import re

import numpy as np
import ot
import vtk
from vtk.util import numpy_support as vnp

from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

DATA_DIR = os.path.abspath("./cloud")
OUT_DIR = os.path.abspath("./cloud_output")
os.makedirs(OUT_DIR, exist_ok=True)

SCALAR_ARRAY = "Scalars_"
PERSISTENCE_THRESHOLD = 0.5
N_TRACK_STEPS = 3
SPHERE_RADIUS = 2.0
MAXIMUM_CRITICAL_TYPE = 3  # ttkScalarFieldCriticalPoints convention (2D field)

# ---------------------------------------------------------------------------
# 1. Load the time-dependent dataset (directory of .vti files -> time series)
# ---------------------------------------------------------------------------
def sort_key(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0

files = sorted(glob.glob(os.path.join(DATA_DIR, "cloud*.vti")), key=sort_key)
print("Found %d timesteps" % len(files))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = [SCALAR_ARRAY]
reader.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [900, 900]

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.AnimationTime = 0

# ---------------------------------------------------------------------------
# 2. Render the original scalar field
# ---------------------------------------------------------------------------
orig_disp = Show(reader, view)
ColorBy(orig_disp, ('POINTS', SCALAR_ARRAY))
orig_disp.SetScalarBarVisibility(view, True)
warmcold = GetColorTransferFunction(SCALAR_ARRAY)
warmcold.ApplyPreset('Warm to Cold', True)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "01_original_scalar_field.png"), view)
print("Saved original scalar field render.")

# ---------------------------------------------------------------------------
# 3. Persistence simplification (threshold = 0.5) on the maxima, and per-
#    timestep persistent-maxima extraction for the first N_TRACK_STEPS steps
# ---------------------------------------------------------------------------
simplify = TTKTopologicalSimplificationByPersistence(Input=reader)
simplify.InputArray = ['POINTS', SCALAR_ARRAY]
simplify.PairType = 'Maximum-Saddle'
simplify.PersistenceThreshold = PERSISTENCE_THRESHOLD
simplify.ThresholdIsAbsolute = 0  # relative persistence threshold
simplify.UpdatePipeline()

simp_disp = Show(simplify, view)
ColorBy(simp_disp, ('POINTS', SCALAR_ARRAY))
simp_disp.SetScalarBarVisibility(view, True)
Hide(reader, view)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "02_simplified_scalar_field.png"), view)
print("Saved simplified (persistence=0.5) scalar field render.")

critical_points = TTKScalarFieldCriticalPoints(Input=simplify)
critical_points.ScalarField = ['POINTS', SCALAR_ARRAY]

maxima_per_step = []  # list of (N_t x 3) coordinate arrays
values_per_step = []  # list of (N_t,) scalar-value arrays

for t in range(N_TRACK_STEPS):
    # Re-read this single timestep on its own pipeline so simplification and
    # critical-point extraction run on exactly the right slice of data.
    step_reader = XMLImageDataReader(FileName=[files[t]])
    step_reader.PointArrayStatus = [SCALAR_ARRAY]
    step_reader.UpdatePipeline()

    step_simp = TTKTopologicalSimplificationByPersistence(Input=step_reader)
    step_simp.InputArray = ['POINTS', SCALAR_ARRAY]
    step_simp.PairType = 'Maximum-Saddle'
    step_simp.PersistenceThreshold = PERSISTENCE_THRESHOLD
    step_simp.ThresholdIsAbsolute = 0
    step_simp.UpdatePipeline()

    step_cp = TTKScalarFieldCriticalPoints(Input=step_simp)
    step_cp.ScalarField = ['POINTS', SCALAR_ARRAY]
    step_cp.UpdatePipeline()

    cp_data = servermanager.Fetch(step_cp)
    n_points = cp_data.GetNumberOfPoints()
    crit_types = vnp.vtk_to_numpy(cp_data.GetPointData().GetArray('CriticalType'))
    scalars = vnp.vtk_to_numpy(cp_data.GetPointData().GetArray(SCALAR_ARRAY))
    coords = vnp.vtk_to_numpy(cp_data.GetPoints().GetData()) if n_points else np.zeros((0, 3))

    max_mask = crit_types == MAXIMUM_CRITICAL_TYPE
    max_coords = coords[max_mask]
    max_values = scalars[max_mask]

    maxima_per_step.append(max_coords)
    values_per_step.append(max_values)
    print("Timestep %d: %d persistent maxima extracted" % (t, max_coords.shape[0]))

# ---------------------------------------------------------------------------
# 4. Track maxima across the first N_TRACK_STEPS timesteps using an Earth
#    Mover's Distance (optimal transport) assignment between consecutive
#    timesteps' maxima point sets (Python Optimal Transport / POT).
# ---------------------------------------------------------------------------
def match_emd(src_coords, dst_coords):
    """Return, for each source maximum, the index of its EMD-matched
    destination maximum (via the argmax of the optimal transport plan)."""
    n, m = src_coords.shape[0], dst_coords.shape[0]
    if n == 0 or m == 0:
        return np.full((n,), -1, dtype=int)
    cost = ot.dist(src_coords, dst_coords, metric='euclidean')
    a = ot.unif(n)
    b = ot.unif(m)
    plan = ot.emd(a, b, cost)  # earth mover's distance transport plan
    return np.argmax(plan, axis=1)

track_points = []   # flattened list of (x, y, z_layer) for every tracked vertex
track_values = []   # corresponding scalar value
track_lines = []    # list of (id_a, id_b) segments of the piecewise-linear tracks

# Give every timestep's maxima its own z-layer purely for visualization so
# the "piecewise-linear" tracks appear as an actual 3D polyline in time.
LAYER_SPACING = 5.0
point_ids_per_step = []
for t in range(N_TRACK_STEPS):
    ids = []
    for (x, y, _), v in zip(maxima_per_step[t], values_per_step[t]):
        ids.append(len(track_points))
        track_points.append((x, y, t * LAYER_SPACING))
        track_values.append(v)
    point_ids_per_step.append(ids)

n_tracks = 0
for t in range(N_TRACK_STEPS - 1):
    src = maxima_per_step[t]
    dst = maxima_per_step[t + 1]
    if src.shape[0] == 0 or dst.shape[0] == 0:
        continue
    matches = match_emd(src, dst)
    for i, j in enumerate(matches):
        if j < 0:
            continue
        a = point_ids_per_step[t][i]
        b = point_ids_per_step[t + 1][j]
        track_lines.append((a, b))
        n_tracks += 1

print("Built %d piecewise-linear tracking segments across %d timesteps via EMD."
      % (n_tracks, N_TRACK_STEPS))

# ---------------------------------------------------------------------------
# Build a vtkPolyData holding the tracked maxima points + connecting lines
# ---------------------------------------------------------------------------
track_poly = vtk.vtkPolyData()
pts = vtk.vtkPoints()
for p in track_points:
    pts.InsertNextPoint(p)
track_poly.SetPoints(pts)

lines = vtk.vtkCellArray()
for a, b in track_lines:
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, a)
    line.GetPointIds().SetId(1, b)
    lines.InsertNextCell(line)
track_poly.SetLines(lines)

value_array = vnp.numpy_to_vtk(np.asarray(track_values, dtype=np.float64))
value_array.SetName(SCALAR_ARRAY)
track_poly.GetPointData().AddArray(value_array)
track_poly.GetPointData().SetActiveScalars(SCALAR_ARRAY)

tracked_source = TrivialProducer(registrationName="TrackedMaxima")
tracked_source.GetClientSideObject().SetOutput(track_poly)
tracked_source.UpdatePipeline()

# ---------------------------------------------------------------------------
# 5. Visualize tracked critical points (spheres, radius=2) + scalar field,
#    warm-cold colormap
# ---------------------------------------------------------------------------
spheres = TTKIcospheresFromPoints(Input=tracked_source)
spheres.Radius = SPHERE_RADIUS
spheres.UpdatePipeline()

scene.AnimationTime = N_TRACK_STEPS - 1
Show(reader, view)
ColorBy(GetDisplayProperties(reader, view), ('POINTS', SCALAR_ARRAY))
GetDisplayProperties(reader, view).SetScalarBarVisibility(view, True)
Hide(simplify, view)

sphere_disp = Show(spheres, view)
ColorBy(sphere_disp, ('POINTS', SCALAR_ARRAY))
sphere_disp.SetScalarBarVisibility(view, True)
warmcold.ApplyPreset('Warm to Cold', True)

track_lines_disp = Show(tracked_source, view)
ColorBy(track_lines_disp, ('POINTS', SCALAR_ARRAY))
track_lines_disp.LineWidth = 3.0

view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "03_tracked_maxima_with_scalar_field.png"), view)
print("Saved tracked-maxima visualization.")

print("Pipeline complete. Outputs written to:", OUT_DIR)
