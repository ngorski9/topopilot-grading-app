from paraview.simple import *
import glob, os

paraview.simple._DisableFirstRenderCameraReset()

# --- Load the time-varying Ionization dataset (series of .vti files) ---
data_dir = "/workspace/Ionization"
files = sorted(glob.glob(os.path.join(data_dir, "Ionization*.vti")),
               key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))

reader = XMLImageDataReader(registrationName="Ionization", FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = [1400, 900]

# ------------------------------------------------------------------
# 1) Visualize the original scalar field with the viridis colormap
# ------------------------------------------------------------------
origDisplay = Show(reader, renderView, 'UniformGridRepresentation')
origDisplay.SetRepresentationType('Volume')
ColorBy(origDisplay, ('POINTS', 'Scalars_'))
origDisplay.RescaleTransferFunctionToDataRange(True)
origDisplay.SetScalarBarVisibility(renderView, True)

scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)
scalarsPWF = GetOpacityTransferFunction('Scalars_')
origDisplay.SetScalarBarVisibility(renderView, True)

renderView.ResetCamera()
renderView.Update()
Render(renderView)
SaveScreenshot("/workspace/ionization_original_viridis.png", renderView,
               ImageResolution=[1400, 900])

Hide(reader, renderView)

# ------------------------------------------------------------------
# 2) Persistence simplification (threshold = 0.1) then contour tree
# ------------------------------------------------------------------
# a) Persistence diagram of the raw scalar field
persistenceDiagram = TTKPersistenceDiagram(registrationName='PersistenceDiagram', Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

# b) Keep only persistence pairs with persistence > 0.1 (drop the diagonal / noisy pairs)
persistenceThreshold = Threshold(registrationName='PersistenceThreshold', Input=persistenceDiagram)
persistenceThreshold.Scalars = ['CELLS', 'Persistence']
persistenceThreshold.UpperThreshold = 0.1
persistenceThreshold.ThresholdMethod = 'Above Upper Threshold'
persistenceThreshold.UpdatePipeline()

# c) Topological simplification driven by the surviving persistence pairs
topoSimplification = TTKTopologicalSimplification(registrationName='TopologicalSimplification',
                                                    Domain=reader,
                                                    Constraints=persistenceThreshold)
topoSimplification.ScalarField = ['POINTS', 'Scalars_']
topoSimplification.VertexIdentifierField = ['POINTS', 'VertexScalarField']
topoSimplification.UpdatePipeline()

# d) Contour tree of the simplified piecewise-linear scalar field
contourTree = TTKContourTree(registrationName='ContourTree', Input=topoSimplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

nodes = FindSource('ContourTree')  # output port 0: nodes
arcs = OutputPort(contourTree, 1)  # output port 1: arcs

# ------------------------------------------------------------------
# 3) Draw the simplified contour tree: edges (radius 1), vertices (radius 2)
#    colored by critical type: max=red, 2-saddle=orange, 1-saddle=white, min=blue
# ------------------------------------------------------------------
arcsDisplay = Show(arcs, renderView, 'GeometryRepresentation')
arcsDisplay.SetRepresentationType('Wireframe')
arcsDisplay.RenderLinesAsTubes = 1
arcsDisplay.LineWidth = 1 * 2  # radius ~1

nodesDisplay = Show(OutputPort(contourTree, 0), renderView, 'GeometryRepresentation')
nodesDisplay.SetRepresentationType('Points')
nodesDisplay.RenderPointsAsSpheres = 1
nodesDisplay.PointSize = 2 * 2  # radius ~2

ColorBy(nodesDisplay, ('POINTS', 'CriticalType'))
critLUT = GetColorTransferFunction('CriticalType')
critLUT.InterpretValuesAsCategories = 1
critLUT.AnnotationsInitialized = 1
# CriticalType: 0=minimum, 1=1-saddle, 2=2-saddle, 3=maximum
critLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
critLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # minimum -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.5, 0.0,   # 2-saddle -> orange
    1.0, 0.0, 0.0,   # maximum -> red
]
critLUT.IndexedOpacities = [1.0, 1.0, 1.0, 1.0]

arcsDisplay.AmbientColor = [0.7, 0.7, 0.7]
arcsDisplay.DiffuseColor = [0.7, 0.7, 0.7]

renderView.ResetCamera()
Render(renderView)
SaveScreenshot("/workspace/ionization_simplified_contour_tree.png", renderView,
               ImageResolution=[1400, 900])

print("Done. Screenshots written to /workspace/ionization_original_viridis.png "
      "and /workspace/ionization_simplified_contour_tree.png")
