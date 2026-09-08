#!/usr/bin/env pvpython
"""
Time-dependent cloud dataset analysis with ParaView + TTK.

Steps:
 1. Load the ./cloud time-series (cloud1.vti .. cloud34.vti) as a temporal vtkImageData source.
 2. Render the original scalar field ("Scalars_").
 3. Persistence simplification (persistence threshold = 0.5) of the maxima-saddle pairs.
 4. Track piecewise-linear maxima over 3 time steps with TTK's Wasserstein
    (= discrete Earth Mover's Distance) matcher.
 5. Visualize the tracked critical points (glyphed as spheres, radius 2) together with
    the scalar field, both using a warm-cold (blue-white-red) colormap.
"""
import glob
import os
import re

from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

DATA_DIR = "/workspace/cloud"
OUT_DIR = "/workspace/out"
os.makedirs(OUT_DIR, exist_ok=True)

SCALAR_ARRAY = "Scalars_"
PERSISTENCE_VALUE = 0.5
SPHERE_RADIUS = 2.0
N_TRACK_STEPS = 3

# ---------------------------------------------------------------------------
# 0. Collect + sort the time-series files
# ---------------------------------------------------------------------------
def frame_index(path):
    m = re.search(r"cloud(\d+)\.vti$", path)
    return int(m.group(1))

all_files = sorted(glob.glob(os.path.join(DATA_DIR, "cloud*.vti")), key=frame_index)
assert all_files, f"No .vti files found in {DATA_DIR}"
track_files = all_files[:N_TRACK_STEPS]
print(f"Found {len(all_files)} time steps; tracking over first {N_TRACK_STEPS}: "
      f"{[os.path.basename(f) for f in track_files]}")

def warm_cold(display, array_name, component=-1):
    """Apply the classic ParaView 'Blue to Red Rainbow'-free warm/cold diverging map."""
    ColorBy(display, ('POINTS', array_name))
    lut = GetColorTransferFunction(array_name)
    lut.ApplyPreset('Cool to Warm', True)
    display.RescaleTransferFunctionToDataRange(True)
    display.SetScalarBarVisibility(GetActiveView(), True)
    return lut

# ---------------------------------------------------------------------------
# 1. Load full time series & render the ORIGINAL scalar field
# ---------------------------------------------------------------------------
reader_full = XMLImageDataReader(FileName=all_files)
reader_full.PointArrayStatus = [SCALAR_ARRAY]

view1 = CreateRenderView()
view1.ViewSize = [900, 900]
disp_orig = Show(reader_full, view1)
disp_orig.Representation = 'Surface'
warm_cold(disp_orig, SCALAR_ARRAY)
ResetCamera(view1)
view1.CameraPosition = [128, 128, 500]
view1.CameraFocalPoint = [128, 128, 0]
view1.CameraViewUp = [0, 1, 0]
Render(view1)
SaveScreenshot(os.path.join(OUT_DIR, "01_original_scalar_field.png"), view1,
                ImageResolution=[900, 900])
print("Saved: 01_original_scalar_field.png (original scalar field)")

# ---------------------------------------------------------------------------
# 2. Persistence simplification (persistence threshold = 0.5) on the maxima
# ---------------------------------------------------------------------------
simplify = TTKTopologicalSimplificationByPersistence(Input=reader_full)
simplify.InputArray = ['POINTS', SCALAR_ARRAY]
simplify.PairType = 'Maximum-Saddle'
simplify.ThresholdIsAbsolute = 0          # threshold expressed relative to function span
simplify.PersistenceThreshold = PERSISTENCE_VALUE

view2 = CreateRenderView()
view2.ViewSize = [900, 900]
disp_simplified = Show(simplify, view2)
disp_simplified.Representation = 'Surface'
warm_cold(disp_simplified, SCALAR_ARRAY)
ResetCamera(view2)
view2.CameraPosition = [128, 128, 500]
view2.CameraFocalPoint = [128, 128, 0]
view2.CameraViewUp = [0, 1, 0]
Render(view2)
SaveScreenshot(os.path.join(OUT_DIR, "02_simplified_scalar_field.png"), view2,
                ImageResolution=[900, 900])
print(f"Saved: 02_simplified_scalar_field.png "
      f"(persistence-simplified, threshold={PERSISTENCE_VALUE})")

# ---------------------------------------------------------------------------
# 3. Track piecewise-linear maxima over 3 time steps using the Wasserstein /
#    Earth Mover's Distance matcher (TTKTrackingFromPersistenceDiagrams)
# ---------------------------------------------------------------------------
# TTKTrackingFromPersistenceDiagrams expects one persistence diagram (a
# vtkUnstructuredGrid, output of TTKPersistenceDiagram) per time step, connected as
# separate input connections (not grouped into a single multiblock). Each diagram is
# restricted to the maximum-saddle pairs and simplified with the same persistence
# threshold (0.5, expressed here in absolute scalar units) used in step 2, so only
# the surviving piecewise-linear maxima are tracked.
data_range = reader_full.GetDataInformation().GetPointDataInformation().GetArrayInformation(SCALAR_ARRAY).GetComponentRange(0)
abs_persistence_threshold = PERSISTENCE_VALUE * (data_range[1] - data_range[0])

maxima_diagrams = []
for f in track_files:
    frame_reader = XMLImageDataReader(FileName=[f])
    frame_reader.PointArrayStatus = [SCALAR_ARRAY]
    diagram = TTKPersistenceDiagram(Input=frame_reader)
    diagram.ScalarField = ['POINTS', SCALAR_ARRAY]

    max_saddle_pairs = Threshold(Input=diagram)
    max_saddle_pairs.Scalars = ['CELLS', 'PairType']
    max_saddle_pairs.ThresholdMethod = 'Between'
    max_saddle_pairs.LowerThreshold = 1   # 1 == saddle-maximum pairs
    max_saddle_pairs.UpperThreshold = 1

    persistent_maxima = Threshold(Input=max_saddle_pairs)
    persistent_maxima.Scalars = ['CELLS', 'Persistence']
    persistent_maxima.ThresholdMethod = 'Between'
    persistent_maxima.LowerThreshold = abs_persistence_threshold
    persistent_maxima.UpperThreshold = 1e30
    persistent_maxima.UpdatePipeline()

    maxima_diagrams.append(persistent_maxima)

tracking = TTKTrackingFromPersistenceDiagrams(Input=maxima_diagrams)
tracking.Persistencethreshold = 0.0   # diagrams are already simplified above
tracking.pparameter = "2"             # Wasserstein-2 == discrete Earth Mover's Distance (L2 ground metric)
# 'Assignmentmethod' defaults to pMunkres/Wasserstein (Earth Mover's Distance) matching.
tracking.UpdatePipeline()

tracked_info = tracking.GetDataInformation()
tpdi = tracked_info.GetPointDataInformation()
print("Tracking output point data arrays:",
      [tpdi.GetArrayInformation(i).GetName() for i in range(tpdi.GetNumberOfArrays())])
print(f"Tracked {tracked_info.GetNumberOfPoints()} maxima instances across "
      f"{N_TRACK_STEPS} time steps (persistence threshold = {PERSISTENCE_VALUE}, "
      f"absolute = {abs_persistence_threshold:.3f})")

# All surviving points are maxima (CriticalType == 3); kept for robustness/clarity.
maxima_only = Threshold(Input=tracking)
maxima_only.Scalars = ['POINTS', 'CriticalType']
maxima_only.ThresholdMethod = 'Between'
maxima_only.LowerThreshold = 3
maxima_only.UpperThreshold = 3

# ---------------------------------------------------------------------------
# 4. Visualize tracked critical points (radius-2 spheres) + scalar field,
#    both with the warm-cold colormap
# ---------------------------------------------------------------------------
view3 = CreateRenderView()
view3.ViewSize = [900, 900]

# background scalar field (last of the 3 tracked frames)
bg_reader = XMLImageDataReader(FileName=[track_files[-1]])
bg_reader.PointArrayStatus = [SCALAR_ARRAY]
disp_bg = Show(bg_reader, view3)
disp_bg.Representation = 'Surface'
warm_cold(disp_bg, SCALAR_ARRAY)

# glyph tracked maxima as spheres of radius 2
glyph = Glyph(Input=maxima_only, GlyphType='Sphere')
glyph.GlyphType.Radius = SPHERE_RADIUS
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'

disp_glyph = Show(glyph, view3)
disp_glyph.Representation = 'Surface'
warm_cold(disp_glyph, 'CriticalType')
disp_glyph.SetScalarBarVisibility(view3, False)  # avoid a duplicate legend

# also show the tracking edges (trajectories) themselves
disp_tracks = Show(tracking, view3)
disp_tracks.Representation = 'Surface'
disp_tracks.LineWidth = 3
warm_cold(disp_tracks, 'CriticalType')

ResetCamera(view3)
view3.CameraPosition = [128, 128, 500]
view3.CameraFocalPoint = [128, 128, 0]
view3.CameraViewUp = [0, 1, 0]
Render(view3)
SaveScreenshot(os.path.join(OUT_DIR, "03_tracked_maxima.png"), view3,
                ImageResolution=[900, 900])
print("Saved: 03_tracked_maxima.png (tracked maxima spheres + scalar field, warm-cold colormap)")

print("\nDone. Outputs written to", OUT_DIR)
