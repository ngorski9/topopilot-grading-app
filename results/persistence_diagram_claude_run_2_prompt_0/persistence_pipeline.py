from paraview.simple import *
import os

# ---------------------------------------------------------------------------
# 1. Load the scalar field
# ---------------------------------------------------------------------------
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

pointData = reader.PointData
if len(pointData.keys()) == 0:
    raise RuntimeError("No point data arrays found in QMCPACK.vti")
scalarName = pointData.keys()[0]
print("Using scalar array:", scalarName)

# TTK generally expects point data; make sure the array is set as active scalar.
reader.PointArrayStatus = [scalarName]

# ---------------------------------------------------------------------------
# 2. Compute the initial persistence diagram (needed to drive simplification)
# ---------------------------------------------------------------------------
persistenceDiagram1 = TTKPersistenceDiagram(Input=reader)
persistenceDiagram1.ScalarField = ['POINTS', scalarName]
persistenceDiagram1.UpdatePipeline()

# ---------------------------------------------------------------------------
# 3. Keep only pairs whose persistence is above the 0.04 threshold
#    (this selection of critical points drives topological simplification)
# ---------------------------------------------------------------------------
threshold = Threshold(Input=persistenceDiagram1)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.04
threshold.UpperThreshold = 1e12
threshold.ThresholdMethod = 'Above Upper Threshold' if False else 'Between'
threshold.UpdatePipeline()

# ---------------------------------------------------------------------------
# 4. Topological simplification of the scalar field using the surviving pairs
# ---------------------------------------------------------------------------
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', scalarName]
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplification.UpdatePipeline()

# ---------------------------------------------------------------------------
# 5. Recompute the persistence diagram on the simplified field
# ---------------------------------------------------------------------------
persistenceDiagram2 = TTKPersistenceDiagram(Input=simplification)
persistenceDiagram2.ScalarField = ['POINTS', scalarName]
persistenceDiagram2.UpdatePipeline()

# ---------------------------------------------------------------------------
# 6. Keep sublevel-set pairs of type 0 (min-saddle) and type 2 (saddle-max)
#    explicitly, excluding type 1 (saddle-saddle in 3D)
# ---------------------------------------------------------------------------
type0 = Threshold(Input=persistenceDiagram2)
type0.Scalars = ['CELLS', 'PairType']
type0.LowerThreshold = 0
type0.UpperThreshold = 0
type0.ThresholdMethod = 'Between'
type0.UpdatePipeline()

type2 = Threshold(Input=persistenceDiagram2)
type2.Scalars = ['CELLS', 'PairType']
type2.LowerThreshold = 2
type2.UpperThreshold = 2
type2.ThresholdMethod = 'Between'
type2.UpdatePipeline()

typeFilter = AppendDatasets(Input=[type0, type2])
typeFilter.UpdatePipeline()

# ---------------------------------------------------------------------------
# 7. Visualize the persistence diagram geometry (TTK encodes it as line
#    segments in the X-Y plane: X = birth, Y = death). Render it directly.
# ---------------------------------------------------------------------------
view = CreateRenderView()
view.ViewSize = [900, 700]
view.InteractionMode = '2D'
view.OrientationAxesVisibility = 0

display = Show(typeFilter, view)
display.Representation = 'Surface'
display.LineWidth = 3
ColorBy(display, ('CELLS', 'PairType'))
display.SetScalarBarVisibility(view, True)

view.ResetCamera()
Render(view)
SaveScreenshot('/workspace/persistence_diagram.png', view, ImageResolution=[900, 700])

print("Saved persistence diagram screenshot to /workspace/persistence_diagram.png")
