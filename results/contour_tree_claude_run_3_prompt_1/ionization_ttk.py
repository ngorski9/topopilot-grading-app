from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

outdir = "/workspace/output"
os.makedirs(outdir, exist_ok=True)

# --- Load the time-varying Ionization dataset ---
files = sorted(
    [os.path.join("/workspace/Ionization", f) for f in os.listdir("/workspace/Ionization") if f.endswith(".vti")],
    key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))))
)
reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]

# ============================================================
# 1) Render the original scalar field with viridis colormap
# ============================================================
origDisplay = Show(reader, view)
origDisplay.SetRepresentationType('Volume')
ColorBy(origDisplay, ('POINTS', 'Scalars_'))
origDisplay.SetScalarBarVisibility(view, True)
origLUT = GetColorTransferFunction('Scalars_')
origLUT.ApplyPreset('Viridis (matplotlib)', True)
origDisplay.RescaleTransferFunctionToDataRange(True)

view.ResetCamera()
Render(view)
SaveScreenshot(os.path.join(outdir, "01_original_scalar_field_viridis.png"), view, ImageResolution=[1200, 900])

Hide(reader, view)

# ============================================================
# 2) Persistence simplification (value 0.1)
# ============================================================
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

# keep only the pairs below the 0.1 persistence threshold: these
# are the ones that get removed by TopologicalSimplification
critPairs = Threshold(Input=persistenceDiagram)
critPairs.Scalars = ['CELLS', 'Persistence']
critPairs.LowerThreshold = 0.0
critPairs.UpperThreshold = 0.1
critPairs.ThresholdMethod = 'Between'
critPairs.UpdatePipeline()

simplification = TTKTopologicalSimplification(Domain=reader, Constraints=critPairs)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplification.UpdatePipeline()

# ============================================================
# 3) Compute the piecewise-linear contour tree of the simplified field
# ============================================================
contourTree = TTKContourTree(Input=simplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

# TTKContourTree exposes 3 output ports: 0 = nodes, 1 = arcs, 2 = segmentation
ctNodes = OutputPort(contourTree, 0)
ctArcs = OutputPort(contourTree, 1)

nodesDisplay = Show(ctNodes, view)
nodesDisplay.Representation = 'Point Gaussian'
nodesDisplay.GaussianRadius = 2
nodesDisplay.ShaderPreset = 'Plain circle'

# Color vertices by CriticalType: 0=min, 1=1-saddle, 2=2-saddle, 3=max
ColorBy(nodesDisplay, ('POINTS', 'CriticalType'))
critLUT = GetColorTransferFunction('CriticalType')
critLUT.InterpretValuesAsCategories = 1
critLUT.AnnotationsInitialized = 1
critLUT.Annotations = ['0', 'min', '1', '1-saddle', '2', '2-saddle', '3', 'max']
critLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # min -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.5, 0.0,   # 2-saddle -> orange
    1.0, 0.0, 0.0,   # max -> red
]
nodesDisplay.SetScalarBarVisibility(view, False)

edgesDisplay = Show(ctArcs, view)
edgesDisplay.Representation = 'Surface'
edgesDisplay.LineWidth = 1
edgesDisplay.AmbientColor = [0.0, 0.0, 0.0]
edgesDisplay.DiffuseColor = [0.0, 0.0, 0.0]
edgesDisplay.SetScalarBarVisibility(view, False)

view.ResetCamera()
Render(view)
SaveScreenshot(os.path.join(outdir, "02_contour_tree_simplified_0.1.png"), view, ImageResolution=[1200, 900])

print("DONE")
