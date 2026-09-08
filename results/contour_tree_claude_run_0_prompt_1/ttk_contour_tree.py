from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# --- Load data (time series) ---
reader = XMLImageDataReader(registrationName='Ionization', FileName=[
    '/workspace/Ionization/Ionization%d.vti' % i for i in range(1, 22)
])
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

# --- View 1: original scalar field with viridis colormap ---
view1 = CreateRenderView()
view1.ViewSize = [900, 800]

disp1 = Show(reader, view1)
disp1.SetRepresentationType('Volume')
ColorBy(disp1, ('POINTS', 'Scalars_'))
disp1.SetScalarBarVisibility(view1, True)
scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)
disp1.RescaleTransferFunctionToDataRange(True)

view1.ResetCamera()
Render(view1)
SaveScreenshot('/workspace/original_scalar_field.png', view1, ImageResolution=[900, 800])

# --- TTK pipeline ---
# 1. Persistence Diagram
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

# 2. Threshold the diagram to keep pairs with persistence >= 0.1
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.1
threshold.UpperThreshold = 0.1
threshold.ThresholdMethod = 'Above Upper Threshold'
threshold.UpdatePipeline()

# 3. Topological Simplification using the simplified persistence pairs
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplification.UpdatePipeline()

# 4. Contour Tree (piecewise-linear) via TTK's FTM-based Contour Tree module
contourTree = TTKContourTree(Input=simplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

# Output port 0: nodes (skeleton nodes), port 1: arcs (skeleton arcs)
ctNodes = OutputPort(contourTree, 0)
ctArcs = OutputPort(contourTree, 1)

# --- View 2: simplified contour tree ---
view2 = CreateRenderView()
view2.ViewSize = [900, 800]

# Edges (arcs)
arcsDisp = Show(ctArcs, view2)
arcsDisp.Representation = 'Surface'
arcsDisp.LineWidth = 1
arcsDisp.DiffuseColor = [0.7, 0.7, 0.7]

# Nodes (vertices) colored by CriticalType
nodesDisp = Show(ctNodes, view2)
nodesDisp.Representation = 'Point Gaussian'
nodesDisp.GaussianRadius = 2
nodesDisp.ShaderPreset = 'Plain circle'

ColorBy(nodesDisp, ('POINTS', 'CriticalType'))
critLUT = GetColorTransferFunction('CriticalType')
critLUT.InterpretValuesAsCategories = 1
critLUT.AnnotationsInitialized = 1

# CriticalType values (3D): 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum
critLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
critLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # minimum -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.5, 0.0,   # 2-saddle -> orange
    1.0, 0.0, 0.0,   # maximum -> red
]
nodesDisp.SetScalarBarVisibility(view2, True)

view2.ResetCamera()
Render(view2)
SaveScreenshot('/workspace/simplified_contour_tree.png', view2, ImageResolution=[900, 800])

print("Done. Screenshots saved to /workspace/original_scalar_field.png and /workspace/simplified_contour_tree.png")
