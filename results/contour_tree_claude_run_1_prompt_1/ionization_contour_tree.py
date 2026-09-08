from paraview.simple import *
from paraview import servermanager
import glob, os

paraview.simple._DisableFirstRenderCameraReset()

files = sorted(glob.glob('/workspace/Ionization/Ionization*.vti'),
               key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

# ---------------------------------------------------------------
# 1. Render the original scalar field with viridis colormap
# ---------------------------------------------------------------
view1 = CreateRenderView()
disp1 = Show(reader, view1)
disp1.Representation = 'Surface'
ColorBy(disp1, ('POINTS', 'Scalars_'))
disp1.RescaleTransferFunctionToDataRange(True)
scalars_LUT = GetColorTransferFunction('Scalars_')
scalars_LUT.ApplyPreset('Viridis (matplotlib)', True)
disp1.SetScalarBarVisibility(view1, True)
view1.ResetCamera()
view1.OrientationAxesVisibility = 0
Render(view1)
SaveScreenshot('/workspace/original_scalar_field.png', view1, ImageResolution=[1024, 768])

# ---------------------------------------------------------------
# 2. TTK pipeline: persistence diagram -> topological simplification
#    -> persistence-simplified merge/contour tree
# ---------------------------------------------------------------
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

# Threshold the persistence diagram pairs by persistence value (0.1)
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.1
threshold.UpperThreshold = 1e9
threshold.ThresholdMethod = 'Above Upper Threshold' if False else 'Between'
threshold.LowerThreshold = 0.1
threshold.UpperThreshold = 1e9
threshold.UpdatePipeline()

topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
topoSimplification.ScalarField = ['POINTS', 'Scalars_']
topoSimplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
topoSimplification.UpdatePipeline()

contourTree = TTKContourTree(Input=topoSimplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

# The filter has two outputs: 0 = skeleton nodes, 1 = skeleton arcs
nodes = OutputPort(contourTree, 0)
arcs = OutputPort(contourTree, 1)

view2 = CreateRenderView()

# ----- Edges -----
edgeDisp = Show(arcs, view2)
edgeDisp.Representation = 'Surface'
edgeDisp.LineWidth = 1
edgeDisp.AmbientColor = [0.0, 0.0, 0.0]
edgeDisp.DiffuseColor = [0.0, 0.0, 0.0]

# ----- Vertices, colored by critical type -----
nodeDisp = Show(nodes, view2)
nodeDisp.Representation = 'Point Gaussian'
nodeDisp.GaussianRadius = 2.0
nodeDisp.ShaderPreset = 'Sphere'

ColorBy(nodeDisp, ('POINTS', 'CriticalType'))
critLUT = GetColorTransferFunction('CriticalType')
critLUT.InterpretValuesAsCategories = 1
critLUT.AnnotationsInitialized = 1

# TTK CriticalType convention: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum
critLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
critLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # minimum -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.5, 0.0,   # 2-saddle -> orange
    1.0, 0.0, 0.0,   # maximum -> red
]
nodeDisp.SetScalarBarVisibility(view2, False)

view2.OrientationAxesVisibility = 0
view2.Background = [0.32, 0.34, 0.43]
view2.ResetCamera()
Render(view2)
SaveScreenshot('/workspace/contour_tree_simplified.png', view2, ImageResolution=[1024, 768])

print('Done. Screenshots written to /workspace/original_scalar_field.png and /workspace/contour_tree_simplified.png')
