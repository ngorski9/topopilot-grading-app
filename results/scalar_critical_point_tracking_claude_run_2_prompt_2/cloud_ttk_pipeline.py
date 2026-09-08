try:
    paraview.simple
except NameError:
    from paraview.simple import *
from paraview import simple

import glob, os

paraview.simple._DisableFirstRenderCameraReset()

datadir = os.path.abspath('./cloud')
files = sorted(glob.glob(os.path.join(datadir, 'cloud*.vti')),
               key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))

# Use the first 3 time steps for the tracking, as requested.
nsteps = 3
files = files[:nsteps]
print("Using files:", files)

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

renderView1 = GetActiveViewOrCreate('RenderView')

# ---------------------------------------------------------------------------
# 1. Show the original scalar field
# ---------------------------------------------------------------------------
origDisplay = Show(reader, renderView1)
ColorBy(origDisplay, ('POINTS', 'Scalars_'))
origDisplay.RescaleTransferFunctionToDataRange(True)
scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Warm to Cold', True)
origDisplay.SetScalarBarVisibility(renderView1, True)
renderView1.ResetCamera()
SaveScreenshot(os.path.abspath('original_scalar_field.png'), renderView1,
               ImageResolution=[1024, 1024])
Hide(reader, renderView1)

# ---------------------------------------------------------------------------
# 2. Persistence simplification (threshold 0.5) via TTK
#    Standard TTK approach:
#      TTKPersistenceDiagram -> Threshold(Persistence>=0.5) -> TTKTopologicalSimplification
# ---------------------------------------------------------------------------
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

# TTK's Persistence field on this dataset is expressed as a percentage of
# the function span (0-100), so a normalized persistence threshold of 0.5
# (50% of the span) is applied as LowerThreshold=50.
PERSISTENCE_THRESHOLD = 0.5
persistenceThreshold = Threshold(Input=persistenceDiagram)
persistenceThreshold.Scalars = ['CELLS', 'Persistence']
persistenceThreshold.LowerThreshold = PERSISTENCE_THRESHOLD * 100
persistenceThreshold.UpperThreshold = 100.0
persistenceThreshold.ThresholdMethod = 'Between'
persistenceThreshold.UpdatePipeline()

topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=persistenceThreshold)
topoSimplification.ScalarField = ['POINTS', 'Scalars_']
topoSimplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
topoSimplification.UpdatePipeline()

simplifiedDisplay = Show(topoSimplification, renderView1)
ColorBy(simplifiedDisplay, ('POINTS', 'Scalars_'))
simplifiedDisplay.RescaleTransferFunctionToDataRange(True)
simplifiedLUT = GetColorTransferFunction('Scalars_')
simplifiedLUT.ApplyPreset('Warm to Cold', True)
simplifiedDisplay.SetScalarBarVisibility(renderView1, True)
renderView1.ResetCamera()
SaveScreenshot(os.path.abspath('simplified_scalar_field.png'), renderView1,
               ImageResolution=[1024, 1024])
Hide(topoSimplification, renderView1)

# ---------------------------------------------------------------------------
# 3. Track piecewise-linear maxima for 3 time steps using Earth Mover's Distance
#    Per timestep: reader -> TTKPersistenceDiagram -> Threshold(0.5) ->
#    TTKTopologicalSimplification -> TTKPersistenceDiagram (simplified).
#    TTKTrackingFromPersistenceDiagrams takes ONE input connection PER
#    timestep (it is not a temporal-loop filter), so each simplified,
#    per-timestep persistence diagram is connected as a separate input.
# ---------------------------------------------------------------------------
perTimestepDiagrams = []
for f in files:
    stepReader = XMLImageDataReader(FileName=[f])
    stepReader.PointArrayStatus = ['Scalars_']
    stepReader.UpdatePipeline()

    stepPD = TTKPersistenceDiagram(Input=stepReader)
    stepPD.ScalarField = ['POINTS', 'Scalars_']
    stepPD.UpdatePipeline()

    stepThreshold = Threshold(Input=stepPD)
    stepThreshold.Scalars = ['CELLS', 'Persistence']
    stepThreshold.LowerThreshold = PERSISTENCE_THRESHOLD * 100
    stepThreshold.UpperThreshold = 100.0
    stepThreshold.ThresholdMethod = 'Between'
    stepThreshold.UpdatePipeline()

    stepSimplification = TTKTopologicalSimplification(Domain=stepReader, Constraints=stepThreshold)
    stepSimplification.ScalarField = ['POINTS', 'Scalars_']
    stepSimplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
    stepSimplification.UpdatePipeline()

    stepSimplifiedPD = TTKPersistenceDiagram(Input=stepSimplification)
    stepSimplifiedPD.ScalarField = ['POINTS', 'Scalars_']
    stepSimplifiedPD.UpdatePipeline()

    perTimestepDiagrams.append(stepSimplifiedPD)

# TTKTrackingFromPersistenceDiagrams matches diagrams between consecutive
# timesteps using the p-Wasserstein (i.e. Earth Mover's Distance) assignment.
tracking = TTKTrackingFromPersistenceDiagrams(Input=perTimestepDiagrams)
tracking.pparameter = '2'             # Wasserstein-2 = Earth Mover's Distance
tracking.Assignmentmethod = 'ttk: pMunkres (Wasserstein), Gabow-Tarjan (Bottleneck)'
# Emphasize maxima over saddles when weighting the matching so the tracked
# trajectories follow the piecewise-linear maxima.
tracking.Extremumweight = 1.0
tracking.Saddleweight = 0.0
tracking.Persistencethreshold = PERSISTENCE_THRESHOLD * 100
tracking.UpdatePipeline()

# ---------------------------------------------------------------------------
# 4. Visualize the tracked critical points together with the scalar field
#    sphere glyphs, radius 2, warm-to-cold colormap
# ---------------------------------------------------------------------------
fieldDisplay = Show(reader, renderView1)
ColorBy(fieldDisplay, ('POINTS', 'Scalars_'))
fieldDisplay.RescaleTransferFunctionToDataRange(True)
fieldLUT = GetColorTransferFunction('Scalars_')
fieldLUT.ApplyPreset('Warm to Cold', True)
fieldDisplay.SetScalarBarVisibility(renderView1, True)
fieldDisplay.Opacity = 0.6

# Show the tracking edges (trajectories) themselves.
trackDisplay = Show(tracking, renderView1)
trackDisplay.SetRepresentationType('Surface')
trackDisplay.LineWidth = 3.0

# Glyph the tracked critical points as spheres of radius 2.
sphereGlyph = Glyph(Input=tracking, GlyphType='Sphere')
sphereGlyph.GlyphType.Radius = 2.0
sphereGlyph.ScaleArray = ['POINTS', 'No scale array']
sphereGlyph.ScaleFactor = 1.0
sphereGlyph.GlyphMode = 'All Points'
sphereGlyph.UpdatePipeline()

glyphDisplay = Show(sphereGlyph, renderView1)
# The tracking output carries TimeStep/CriticalType/ConnectedComponentId
# point data (not the original "Scalars_" field) -- color the tracked
# maxima by timestep so their evolution across the 3 steps is visible.
ColorBy(glyphDisplay, ('POINTS', 'TimeStep'))
glyphDisplay.RescaleTransferFunctionToDataRange(True)
trackLUT = GetColorTransferFunction('TimeStep')
trackLUT.ApplyPreset('Warm to Cold', True)
glyphDisplay.SetScalarBarVisibility(renderView1, True)

renderView1.ResetCamera()
SaveScreenshot(os.path.abspath('tracked_maxima_with_field.png'), renderView1,
               ImageResolution=[1024, 1024])

print("Done. Screenshots written: original_scalar_field.png, "
      "simplified_scalar_field.png, tracked_maxima_with_field.png")
