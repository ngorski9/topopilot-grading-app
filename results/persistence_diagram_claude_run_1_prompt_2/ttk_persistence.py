from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

INPUT = "/workspace/QMCPACK.vti"
ARRAY = "Scalars_"
PERSISTENCE_THRESHOLD = 0.04

# 1. Load data
reader = XMLImageDataReader(FileName=[INPUT])
reader.PointArrayStatus = [ARRAY]

# TTK filters generally need vtkImageData -> fine, but many TTK filters
# require an explicit triangulation-friendly dataset; XMLImageDataReader output works.

# 2. Initial Persistence Diagram (used to build simplification constraints)
pd0 = TTKPersistenceDiagram(Input=reader)
pd0.ScalarField = ['POINTS', ARRAY]
pd0.EmbedinDomain = 1

# 3. Threshold the diagram pairs by persistence >= 0.04 to keep as constraints
#    (Persistence is stored in the 'Persistence' array on the diagram)
thresh = Threshold(Input=pd0)
thresh.Scalars = ['CELLS', 'Persistence']
thresh.LowerThreshold = PERSISTENCE_THRESHOLD
thresh.UpperThreshold = 1e10
thresh.ThresholdMethod = 'Between'

# 4. Topological Simplification using the thresholded diagram as constraints
simplify = TTKTopologicalSimplification(Domain=reader, Constraints=thresh)
simplify.ScalarField = ['POINTS', ARRAY]
simplify.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']

# 5. Recompute the persistence diagram (piecewise-linear sub-level set filtration)
#    on the simplified scalar field
pdFinal = TTKPersistenceDiagram(Input=simplify)
pdFinal.ScalarField = ['POINTS', ARRAY]
pdFinal.EmbedinDomain = 0

UpdatePipeline(time=0, proxy=pdFinal)

# 6. Keep only H0 and H2 pairs (PairIdentifier / CriticalType based selection uses
#    the 'PairType' or dimension array; TTK stores it as 'PairIdentifier' plus
#    a "PairType"/"ttkVertexScalarField" — the dimension is under 'PairType' in
#    older versions, in newer ones it's under the same "Persistence" cell data
#    along with a paired "birth"/"death" CriticalType; the homology dimension of
#    each pair is stored in the cell data array named 'PairIdentifier's sibling
#    called 'PairType' (0 = min-saddle -> H0, 2 = saddle-max -> H2 in 3D).
dim0 = Threshold(Input=pdFinal)
dim0.Scalars = ['CELLS', 'PairType']
dim0.LowerThreshold = 0
dim0.UpperThreshold = 0
dim0.ThresholdMethod = 'Between'

dim2 = Threshold(Input=pdFinal)
dim2.Scalars = ['CELLS', 'PairType']
dim2.LowerThreshold = 2
dim2.UpperThreshold = 2
dim2.ThresholdMethod = 'Between'

merged = AppendDatasets(Input=[dim0, dim2])
UpdatePipeline(time=0, proxy=merged)

# 7. Render the persistence diagram in the birth/death plane (x=birth, y=death)
#    plus the y=x diagonal, using a 2D render view looking down the Z axis.
view = CreateRenderView()
view.InteractionMode = '2D'
view.OrientationAxesVisibility = 0

disp = Show(merged, view)
disp.Representation = 'Points'
disp.PointSize = 8
ColorBy(disp, ('CELLS', 'PairType'))
disp.RescaleTransferFunctionToDataRange(True)
disp.SetScalarBarVisibility(view, True)

diagLine = Line(Point1=[0.0, 0.0, 0.0], Point2=[1.0, 1.0, 0.0])
diagDisp = Show(diagLine, view)
diagDisp.LineWidth = 1
diagDisp.AmbientColor = [1.0, 1.0, 1.0]
diagDisp.DiffuseColor = [1.0, 1.0, 1.0]

view.ResetCamera()
view.CameraPosition = [0.5, 0.5, 1.0]
view.CameraFocalPoint = [0.5, 0.5, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.Update()
Render(view)

SaveScreenshot('/workspace/persistence_diagram_H0_H2.png', view, ImageResolution=[1200, 900])

print("Saved /workspace/persistence_diagram_H0_H2.png")
