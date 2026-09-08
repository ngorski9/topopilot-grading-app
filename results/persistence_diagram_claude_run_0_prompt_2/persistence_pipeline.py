from paraview.simple import *
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 12

SIMPLIFICATION_THRESHOLD = 0.04
SCALAR_FIELD = "Scalars_"

# 1. Read the scalar field
reader = XMLImageDataReader(FileName=["QMCPACK.vti"])
reader.PointArrayStatus = [SCALAR_FIELD]
reader.UpdatePipeline()

# 2. Initial (unsimplified) persistence diagram, sub-level set (ascending) filtration
pd0 = TTKPersistenceDiagram(Input=reader)
pd0.ScalarField = ['POINTS', SCALAR_FIELD]
pd0.IgnoreBoundary = 0
pd0.UpdatePipeline()

# 3. Threshold the diagram: keep only pairs with Persistence >= 0.04, and drop the
#    diagonal, to build the "constraint" set used to simplify the input scalar field
threshold = Threshold(Input=pd0)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = SIMPLIFICATION_THRESHOLD
threshold.ThresholdMethod = 'Above Upper Threshold'
threshold.UpperThreshold = SIMPLIFICATION_THRESHOLD
threshold.UpdatePipeline()

# 4. Topologically simplify the scalar field so that only persistence >= 0.04
#    critical point pairs remain
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', SCALAR_FIELD]
simplification.UpdatePipeline()

# 5. Recompute the persistence diagram on the simplified field: this is the final,
#    persistence-simplified (0.04) piecewise-linear sub-level set filtration diagram
pdFinal = TTKPersistenceDiagram(Input=simplification)
pdFinal.ScalarField = ['POINTS', SCALAR_FIELD]
pdFinal.IgnoreBoundary = 0
pdFinal.UpdatePipeline()

# 6. Keep only H0 (min-saddle, PairType==0) and H2 (saddle-max, PairType==2) pairs
#    (drop H1 saddle-saddle pairs and the diagonal, PairType==-1)
h0h2 = Threshold(Input=pdFinal)
h0h2.Scalars = ['CELLS', 'PairType']
h0h2.LowerThreshold = 0
h0h2.UpperThreshold = 2
h0h2.ThresholdMethod = 'Between'
h0h2.AllScalars = 1
h0h2.UpdatePipeline()

# Actually 'Between' includes PairType==1 (H1); explicitly select {0,2} via two thresholds + merge
h0 = Threshold(Input=pdFinal)
h0.Scalars = ['CELLS', 'PairType']
h0.LowerThreshold = 0
h0.UpperThreshold = 0
h0.ThresholdMethod = 'Between'
h0.UpdatePipeline()

h2 = Threshold(Input=pdFinal)
h2.Scalars = ['CELLS', 'PairType']
h2.LowerThreshold = 2
h2.UpperThreshold = 2
h2.ThresholdMethod = 'Between'
h2.UpdatePipeline()

diagramH0H2 = MergeBlocks(Input=GroupDatasets(Input=[h0, h2]))
diagramH0H2.UpdatePipeline()

SaveData("/workspace/persistence_diagram_H0_H2.vtu", proxy=diagramH0H2)
SaveData("/workspace/persistence_diagram_full.vtu", proxy=pdFinal)

print("H0 pairs:", h0.GetDataInformation().GetNumberOfCells())
print("H2 pairs:", h2.GetDataInformation().GetNumberOfCells())
print("Total simplified pairs:", pdFinal.GetDataInformation().GetNumberOfCells())

# 7. Render the persistence diagram (birth/death embedding produced by TTK)
view = CreateRenderView()
view.ViewSize = [1000, 800]
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]

disp = Show(diagramH0H2, view)
ColorBy(disp, ('CELLS', 'PairType'))
disp.SetScalarBarVisibility(view, True)
disp.LineWidth = 3
disp.RenderLinesAsTubes = 1

# also show full diagonal/all pairs faintly for context
dispFull = Show(pdFinal, view)
dispFull.Representation = 'Wireframe'
dispFull.AmbientColor = [0.8, 0.8, 0.8]
dispFull.DiffuseColor = [0.8, 0.8, 0.8]
dispFull.Opacity = 0.3

view.CameraParallelProjection = 1
view.InteractionMode = '2D'
bounds = pdFinal.GetDataInformation().GetBounds()
cx = 0.5 * (bounds[0] + bounds[1])
cy = 0.5 * (bounds[2] + bounds[3])
view.CameraFocalPoint = [cx, cy, 0]
view.CameraPosition = [cx, cy, 1]
view.CameraViewUp = [0, 1, 0]
ResetCamera(view)
view.CameraPosition = [cx, cy, 1]
view.CameraFocalPoint = [cx, cy, 0]
view.CameraViewUp = [0, 1, 0]

SaveScreenshot("/workspace/persistence_diagram_H0_H2.png", view, ImageResolution=[1000, 800])
print("Done. Screenshot saved to /workspace/persistence_diagram_H0_H2.png")
