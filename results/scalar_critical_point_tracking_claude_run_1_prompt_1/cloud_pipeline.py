"""
TTK pipeline on the time-dependent 'cloud' dataset:
  1. Render the original scalar field.
  2. Persistence simplification (threshold = 0.5).
  3. Track piecewise-linear maxima over 3 time steps with the
     Wasserstein (earth mover's) distance.
  4. Visualize tracked critical points (sphere radius = 2) + scalar field,
     warm-cold colormap.
"""
import glob
import os
import re

from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

OUT_DIR = "/workspace/cloud_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the TTK plugin
# ---------------------------------------------------------------------------
LoadPlugin(
    "/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so",
    remote=False,
    ns=globals(),
)

# ---------------------------------------------------------------------------
# Load the time-dependent dataset (./cloud/cloud*.vti)
# ---------------------------------------------------------------------------
def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

files = sorted(glob.glob("/workspace/cloud/cloud*.vti"), key=natural_key)
print("Found %d timesteps" % len(files))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ["Scalars_"]
reader.UpdatePipeline()

anim = GetAnimationScene()
anim.UpdateAnimationUsingDataTimeSteps()
tsteps = reader.TimestepValues
print("Timesteps:", tsteps[:5], "...")

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1000, 1000]
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]

# ---------------------------------------------------------------------------
# 1) Render the ORIGINAL scalar field (first timestep)
# ---------------------------------------------------------------------------
SetActiveSource(reader)
scene_time_keeper = GetTimeKeeper()
scene_time_keeper.Time = tsteps[0]

disp = Show(reader, view)
ColorBy(disp, ("POINTS", "Scalars_"))
disp.SetScalarBarVisibility(view, True)
disp.RescaleTransferFunctionToDataRange(True)
lut = GetColorTransferFunction("Scalars_")
lut.ApplyPreset("Warm to Cool", True)

view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "01_original_scalar_field.png"), view)
print("Saved 01_original_scalar_field.png")

# ---------------------------------------------------------------------------
# 2) Persistence simplification (threshold = 0.5) on the current timestep
# ---------------------------------------------------------------------------
def build_simplified(source, persistence_threshold=0.5):
    pdiag = TTKPersistenceDiagram(Input=source)
    pdiag.ScalarField = ["POINTS", "Scalars_"]
    pdiag.UpdatePipeline()

    # keep only persistence pairs above the threshold (drop the diagonal too)
    thresh = Threshold(Input=pdiag)
    thresh.Scalars = ["CELLS", "Persistence"]
    thresh.LowerThreshold = persistence_threshold
    thresh.UpperThreshold = 1e10
    thresh.ThresholdMethod = "Between"
    thresh.UpdatePipeline()

    simplification = TTKTopologicalSimplification(
        Domain=source, Constraints=thresh
    )
    simplification.ScalarField = ["POINTS", "Scalars_"]
    simplification.VertexIdentifierField = ["POINTS", "ttkVertexScalarField"]
    simplification.UpdatePipeline()
    return simplification


simplified_t0 = build_simplified(reader, 0.5)

disp2 = Show(simplified_t0, view)
ColorBy(disp2, ("POINTS", "Scalars_"))
disp2.SetScalarBarVisibility(view, True)
disp2.RescaleTransferFunctionToDataRange(True)
lut2 = GetColorTransferFunction("Scalars_")
lut2.ApplyPreset("Warm to Cool", True)
Hide(reader, view)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "02_simplified_scalar_field.png"), view)
print("Saved 02_simplified_scalar_field.png")

# ---------------------------------------------------------------------------
# 3) Track piecewise-linear MAXIMA over 3 time steps (Wasserstein / EMD)
# ---------------------------------------------------------------------------
n_tracking_steps = min(3, len(tsteps))
maxima_diagram_blocks = []

for i in range(n_tracking_steps):
    scene_time_keeper.Time = tsteps[i]
    reader.UpdatePipeline(tsteps[i])

    simplified = build_simplified(reader, 0.5)

    pdiag_simplified = TTKPersistenceDiagram(Input=simplified)
    pdiag_simplified.ScalarField = ["POINTS", "Scalars_"]
    pdiag_simplified.UpdatePipeline(tsteps[i])

    # Keep only saddle-maximum pairs (PairType == 1), i.e. PL maxima,
    # and drop the diagonal (Persistence == 0 pairs use PairType == -1)
    max_pairs = Threshold(Input=pdiag_simplified)
    max_pairs.Scalars = ["CELLS", "PairType"]
    max_pairs.LowerThreshold = 1
    max_pairs.UpperThreshold = 1
    max_pairs.ThresholdMethod = "Between"
    max_pairs.UpdatePipeline(tsteps[i])

    mb = MergeBlocks(Input=max_pairs)
    mb.UpdatePipeline(tsteps[i])
    maxima_diagram_blocks.append(mb)

# TTKTrackingFromPersistenceDiagrams has a repeatable input port: connect
# each per-timestep maxima diagram directly (no manual grouping needed)
tracking = TTKTrackingFromPersistenceDiagrams(Input=maxima_diagram_blocks)
tracking.Assignmentmethod = "ttk: pMunkres (Wasserstein), Gabow-Tarjan (Bottleneck)"
tracking.Persistencethreshold = 0.5
tracking.Extremumweight = 1.0
tracking.Saddleweight = 0.0
tracking.UpdatePipeline()

print("Tracking filter produced:", tracking.GetDataInformation().GetNumberOfPoints(), "points")

# ---------------------------------------------------------------------------
# 4) Visualize tracked critical points (spheres, radius 2) + scalar field
# ---------------------------------------------------------------------------
scene_time_keeper.Time = tsteps[0]
reader.UpdatePipeline(tsteps[0])

field_disp = Show(reader, view)
ColorBy(field_disp, ("POINTS", "Scalars_"))
field_disp.SetScalarBarVisibility(view, True)
field_disp.RescaleTransferFunctionToDataRange(True)
field_lut = GetColorTransferFunction("Scalars_")
field_lut.ApplyPreset("Warm to Cool", True)

glyph = Glyph(Input=tracking, GlyphType="Sphere")
glyph.GlyphType.Radius = 2.0
glyph.ScaleArray = ["POINTS", "No scale array"]
glyph.ScaleFactor = 1.0
glyph.GlyphMode = "All Points"

glyph_disp = Show(glyph, view)
ColorBy(glyph_disp, ("POINTS", "Scalars_"))
glyph_disp.RescaleTransferFunctionToDataRange(True)
glyph_lut = GetColorTransferFunction("Scalars_")
glyph_lut.ApplyPreset("Warm to Cool", True)
glyph_disp.SetScalarBarVisibility(view, True)

view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "03_tracked_maxima.png"), view)
print("Saved 03_tracked_maxima.png")

print("Done. Outputs in", OUT_DIR)
