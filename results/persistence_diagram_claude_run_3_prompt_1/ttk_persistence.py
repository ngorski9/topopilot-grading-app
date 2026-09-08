from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

scalarField = "Scalars_"
persistenceThreshold = 0.04

# 1. Load the data
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.PointArrayStatus = [scalarField]

# 2. Compute the initial persistence diagram (sublevel set filtration)
diagram = TTKPersistenceDiagram(Input=reader)
diagram.ScalarField = ['POINTS', scalarField]
diagram.EmbedinDomain = 0
UpdatePipeline(time=0.0, proxy=diagram)

# 3. Threshold the diagram to keep only pairs with Persistence > persistenceThreshold
#    (this excludes the global min-max pair with a Threshold on Persistence)
diagramThreshold = Threshold(Input=diagram)
diagramThreshold.Scalars = ['CELLS', 'Persistence']
diagramThreshold.LowerThreshold = persistenceThreshold
diagramThreshold.UpperThreshold = 1e9
diagramThreshold.ThresholdMethod = 'Between'
UpdatePipeline(time=0.0, proxy=diagramThreshold)

# 4. Topological simplification using the thresholded diagram as constraints
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=diagramThreshold)
simplification.ScalarField = ['POINTS', scalarField]
UpdatePipeline(time=0.0, proxy=simplification)

# 5. Recompute the persistence diagram on the simplified scalar field
simplifiedDiagram = TTKPersistenceDiagram(Input=simplification)
simplifiedDiagram.ScalarField = ['POINTS', scalarField]
simplifiedDiagram.EmbedinDomain = 0
UpdatePipeline(time=0.0, proxy=simplifiedDiagram)

# 6. Keep only persistence pairs of dimension (order) 0 and 2
# PairType: 0 = min-saddle pairs, 1 = saddle-saddle pairs, 2 = saddle-max pairs (3D), -1 = global min-max pair
dim0 = Threshold(Input=simplifiedDiagram)
dim0.Scalars = ['CELLS', 'PairType']
dim0.LowerThreshold = 0
dim0.UpperThreshold = 0
dim0.ThresholdMethod = 'Between'
UpdatePipeline(time=0.0, proxy=dim0)

dim2 = Threshold(Input=simplifiedDiagram)
dim2.Scalars = ['CELLS', 'PairType']
dim2.LowerThreshold = 2
dim2.UpperThreshold = 2
dim2.ThresholdMethod = 'Between'
UpdatePipeline(time=0.0, proxy=dim2)

appended = AppendDatasets(Input=[dim0, dim2])
UpdatePipeline(time=0.0, proxy=appended)

# 7. Display as a piecewise-linear persistence diagram (2D view: birth vs death, colored by persistence)
view = GetActiveViewOrCreate('RenderView')
view.OrientationAxesVisibility = 0
RenderAllViews()

display = Show(appended, view)
display.Representation = 'Surface'
display.LineWidth = 2.0
ColorBy(display, ('CELLS', 'Persistence'))
display.SetScalarBarVisibility(view, True)

view.ResetCamera()
# Look down the Z axis so birth (x) vs death (y) is shown as a flat 2D diagram
view.CameraPosition = [view.CameraFocalPoint[0], view.CameraFocalPoint[1], view.CameraFocalPoint[2] + 1]
view.CameraViewUp = [0, 1, 0]
view.ResetCamera()
Render()

SaveScreenshot('/workspace/persistence_diagram.png', view, ImageResolution=[1200, 900])
print("Saved persistence diagram screenshot to /workspace/persistence_diagram.png")
