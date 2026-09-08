from paraview.simple import *
import glob

files = sorted(glob.glob('/workspace/Ionization/Ionization*.vti'),
               key=lambda f: int(''.join(filter(str.isdigit, f.split('/')[-1]))))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]

# --- 1. Original scalar field with viridis ---
disp = Show(reader, view)
disp.SetRepresentationType('Volume')
ColorBy(disp, ('POINTS', 'Scalars_'))
disp.SetScalarBarVisibility(view, True)
scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)
ResetCamera(view)
SaveScreenshot('/workspace/original_field_viridis.png', view, ImageResolution=[1200, 900])

Hide(reader, view)

# --- 2. TTK pipeline: PersistenceDiagram -> Threshold -> TopologicalSimplification -> ContourTreeFTM ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.1
threshold.UpperThreshold = 1e9
threshold.ThresholdMethod = 'Above Upper Threshold' if False else 'Between'
try:
    threshold.LowerThreshold = 0.1
    threshold.UpperThreshold = 1e9
except Exception:
    pass
threshold.UpdatePipeline()

simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplification.UpdatePipeline()

contourTree = TTKContourTree(Input=simplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

# The FTM Tree filter has two outputs: nodes (output port 0) and arcs (output port 1)
nodes = OutputPort(contourTree, 0)
arcs = OutputPort(contourTree, 1)

nodesDisp = Show(nodes, view)
nodesDisp.SetRepresentationType('Points')
nodesDisp.PointSize = 2 * 2  # radius 2 -> render as point size

ColorBy(nodesDisp, ('POINTS', 'CriticalType'))
nodesDisp.SetScalarBarVisibility(view, False)

ctLUT = GetColorTransferFunction('CriticalType')
ctLUT.InterpretValuesAsCategories = 1
ctLUT.AnnotationsInitialized = 1
# CriticalType values: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum
ctLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
ctLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # minimum -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.5, 0.0,   # 2-saddle -> orange
    1.0, 0.0, 0.0,   # maximum -> red
]

arcsDisp = Show(arcs, view)
arcsDisp.SetRepresentationType('Surface')
ColorBy(arcsDisp, None)
arcsDisp.LineWidth = 1
arcsDisp.AmbientColor = [0.0, 0.0, 0.0]
arcsDisp.DiffuseColor = [0.0, 0.0, 0.0]

ResetCamera(view)
Render(view)
SaveScreenshot('/workspace/contour_tree_simplified.png', view, ImageResolution=[1200, 900])

print("Done. Saved original_field_viridis.png and contour_tree_simplified.png")
