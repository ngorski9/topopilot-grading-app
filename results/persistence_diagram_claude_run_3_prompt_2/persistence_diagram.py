from paraview.simple import *
from paraview import servermanager as sm

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

SCALAR = 'Scalars_'

# --- Step 1: initial persistence diagram (to drive simplification) ---
pdInitial = TTKPersistenceDiagram(Input=reader)
pdInitial.ScalarField = SCALAR
pdInitial.EmbedinDomain = 0
pdInitial.UpdatePipeline()

# Keep only pairs with Persistence >= 0.04 (this is what gets used to guide
# topological simplification of the scalar field)
critPairs = Threshold(Input=pdInitial)
critPairs.Scalars = ['POINTS', 'Persistence']
critPairs.LowerThreshold = 0.04
critPairs.UpperThreshold = 1e9
critPairs.ThresholdMethod = 'Between'
critPairs.UpdatePipeline()

# --- Step 2: topological simplification of the scalar field using the
#     surviving (persistence >= 0.04) critical point pairs ---
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=critPairs)
simplification.ScalarField = SCALAR
simplification.VertexIdentifierField = 'VertexIdentifier'
simplification.UpdatePipeline()

# --- Step 3: recompute the persistence diagram on the simplified field
#     -> this is the persistence-simplified (persistence >= 0.04) diagram ---
pdFinal = TTKPersistenceDiagram(Input=simplification)
pdFinal.ScalarField = SCALAR
pdFinal.EmbedinDomain = 0
pdFinal.UpdatePipeline()

# Drop any residual pairs below the simplification threshold (numerical noise)
pdSimplified = Threshold(Input=pdFinal)
pdSimplified.Scalars = ['POINTS', 'Persistence']
pdSimplified.LowerThreshold = 0.04
pdSimplified.UpperThreshold = 1e9
pdSimplified.ThresholdMethod = 'Between'
pdSimplified.UpdatePipeline()

# --- Step 4: keep only H0 (min-saddle) and H2 (saddle-max) pairs.
#     TTK encodes the pair dimension in the 'PairType' point-data array
#     (0 = minimum-saddle pair -> H0, 2 = saddle-maximum pair -> H2 in 3D).
pdH0H2 = Threshold(Input=pdSimplified)
pdH0H2.Scalars = ['POINTS', 'PairType']
pdH0H2.LowerThreshold = 0.0
pdH0H2.UpperThreshold = 2.0
pdH0H2.ThresholdMethod = 'Between'
pdH0H2.AllScalars = 0
pdH0H2.UpdatePipeline()

info = pdH0H2.GetDataInformation()
print('Number of H0/H2 persistence pairs after simplification:', info.GetNumberOfCells())

# --- Render the persistence diagram (piecewise-linear diagonal + pairs) ---
# With EmbedinDomain=0 each pair is a 2-point line placed at (birth,0,0)-(death,death,0)
# (TTK's standard 2D persistence-diagram layout) - view it orthographically from +Z.
view = CreateRenderView()
Hide(reader, view)

display = Show(pdH0H2, view)
ColorBy(display, ('POINTS', 'PairType'))
display.SetScalarBarVisibility(view, True)
display.LineWidth = 3.0
display.RenderPointsAsSpheres = True
display.PointSize = 8.0

view.InteractionMode = '2D'
view.CameraParallelProjection = 1
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.Background2 = [1, 1, 1]
view.AxesGrid.Visibility = 1
view.AxesGrid.XTitle = 'Birth'
view.AxesGrid.YTitle = 'Death'
view.AxesGrid.GridColor = [0.0, 0.0, 0.0]

view.ResetCamera()
camera = view.GetActiveCamera()
camera.SetPosition(camera.GetFocalPoint()[0], camera.GetFocalPoint()[1], camera.GetFocalPoint()[2] + 1)
camera.SetViewUp(0, 1, 0)
view.ResetCamera()

Render(view)
SaveScreenshot('/workspace/persistence_diagram.png', view, ImageResolution=[1200, 900])

# Also export the raw diagram (birth/death/persistence/type) as CSV for inspection
SaveData('/workspace/persistence_diagram_H0_H2.csv', proxy=pdH0H2)

print('Wrote /workspace/persistence_diagram.png and /workspace/persistence_diagram_H0_H2.csv')
