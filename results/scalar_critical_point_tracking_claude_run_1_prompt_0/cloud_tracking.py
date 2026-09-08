import os
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud")
SCALAR_ARRAY = "Scalars_"
PERSISTENCE_THRESHOLD = 0.5
SPHERE_RADIUS = 2.0
N_TIMESTEPS = 3
Z_SPACING = 60.0  # visual separation between stacked time steps

# ---------------------------------------------------------------------------
# 1. Load the time-varying cloud dataset
# ---------------------------------------------------------------------------
files = [os.path.join(DATA_DIR, f"cloud{i}.vti") for i in range(1, N_TIMESTEPS + 1)]
reader = OpenDataFile(files)
reader.PointArrayStatus = [SCALAR_ARRAY]
reader.UpdatePipeline()
print("Loaded timesteps:", reader.TimestepValues)

# ---------------------------------------------------------------------------
# 2. Visualize the original scalar field (first time step, animatable)
# ---------------------------------------------------------------------------
renderView = GetActiveViewOrCreate("RenderView")
readerDisplay = Show(reader, renderView)
readerDisplay.Representation = "Surface"
ColorBy(readerDisplay, ("POINTS", SCALAR_ARRAY))
scalarsLUT = GetColorTransferFunction(SCALAR_ARRAY)
scalarsLUT.ApplyPreset("Warm to Cold", True)
readerDisplay.SetScalarBarVisibility(renderView, True)
renderView.ResetCamera()
SaveScreenshot(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_original_scalar_field.png"),
    renderView,
    ImageResolution=[1024, 1024],
)
Hide(reader, renderView)

# ---------------------------------------------------------------------------
# 3-4. Per time step: persistence-simplify (threshold = 0.5) the scalar
#      field, extract its persistence diagram, and stack a visual copy of
#      the simplified field along Z so the 3 time steps can be seen at once.
# ---------------------------------------------------------------------------
diagrams = []
for i in range(N_TIMESTEPS):
    stepReader = OpenDataFile([files[i]])
    stepReader.PointArrayStatus = [SCALAR_ARRAY]

    simplification = TTKTopologicalSimplificationByPersistence(Input=stepReader)
    simplification.InputArray = ["POINTS", SCALAR_ARRAY]
    simplification.PersistenceThreshold = PERSISTENCE_THRESHOLD
    simplification.PairType = "Extremum-Saddle"
    simplification.UpdatePipeline()

    diagram = TTKPersistenceDiagram(Input=simplification)
    diagram.ScalarField = ["POINTS", SCALAR_ARRAY]
    diagram.UpdatePipeline()
    diagrams.append(diagram)

    stacked = Transform(Input=simplification)
    stacked.Transform = "Transform"
    stacked.Transform.Translate = [0.0, 0.0, i * Z_SPACING]
    stacked.UpdatePipeline()

    stackedDisplay = Show(stacked, renderView)
    stackedDisplay.Representation = "Surface"
    ColorBy(stackedDisplay, ("POINTS", SCALAR_ARRAY))
    stackedLUT = GetColorTransferFunction(SCALAR_ARRAY)
    stackedLUT.ApplyPreset("Warm to Cold", True)
    stackedDisplay.SetScalarBarVisibility(renderView, False)

# ---------------------------------------------------------------------------
# 5. Track the piecewise-linear maxima across the 3 time steps using an
#    Earth Mover's Distance (Wasserstein order p=1) match between the
#    persistence-simplified diagrams of consecutive time steps.
# ---------------------------------------------------------------------------
tracking = TTKTrackingFromPersistenceDiagrams(Input=diagrams)
tracking.Persistencethreshold = PERSISTENCE_THRESHOLD
tracking.pparameter = "1"  # Wasserstein order 1 == Earth Mover's Distance
tracking.Assignmentmethod = "ttk: pMunkres (Wasserstein), Gabow-Tarjan (Bottleneck)"
tracking.ForceZtranslation = 1
tracking.ZTranslation = Z_SPACING
tracking.UpdatePipeline()

# the tracked mesh only carries the matched scalar value as cell data
# ("Scalar"); promote it to point data so it can drive the glyph coloring.
trackingPointData = CellDatatoPointData(Input=tracking)
trackingPointData.UpdatePipeline()

# ---------------------------------------------------------------------------
# 6. Display the tracked critical points together with the scalar field:
#    spheres of radius 2, warm-cold colormap.
# ---------------------------------------------------------------------------
COLOR_ARRAY = "Scalar"
glyph = Glyph(Input=trackingPointData, GlyphType="Sphere")
glyph.GlyphType.Radius = SPHERE_RADIUS
glyph.ScaleArray = ["POINTS", "No scale array"]
glyph.ScaleFactor = 1.0
glyph.GlyphMode = "All Points"
glyph.UpdatePipeline()

glyphDisplay = Show(glyph, renderView)
glyphDisplay.Representation = "Surface"
ColorBy(glyphDisplay, ("POINTS", COLOR_ARRAY))
glyphLUT = GetColorTransferFunction(COLOR_ARRAY)
glyphLUT.ApplyPreset("Warm to Cold", True)
glyphDisplay.SetScalarBarVisibility(renderView, True)

trackingDisplay = Show(trackingPointData, renderView)
trackingDisplay.Representation = "Surface"
trackingDisplay.LineWidth = 3.0
ColorBy(trackingDisplay, ("POINTS", COLOR_ARRAY))

renderView.ResetCamera()
renderView.CameraPosition = [900, -900, 600]
renderView.CameraFocalPoint = [128, 128, Z_SPACING]
renderView.CameraViewUp = [0, 0, 1]
renderView.ResetCamera()

outPng = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_tracked_maxima.png")
SaveScreenshot(outPng, renderView, ImageResolution=[1200, 1200])
print("Saved:", outPng)
