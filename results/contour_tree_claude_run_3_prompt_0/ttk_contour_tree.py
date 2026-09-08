from paraview.simple import *
import glob, os

LoadPlugin("/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so", remote=True, ns=globals())

outdir = "/workspace/output"
os.makedirs(outdir, exist_ok=True)

files = sorted(glob.glob("/workspace/Ionization/Ionization*.vti"),
                key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.AnimationTime = scene.TimeKeeper.TimestepValues[0]

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]

# 1) Original scalar field with viridis colormap
origDisplay = Show(reader, view)
ColorBy(origDisplay, ('POINTS', 'Scalars_'))
origDisplay.RescaleTransferFunctionToDataRange(True)
scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)
origDisplay.SetRepresentationType('Volume')
origDisplay.SetScalarBarVisibility(view, True)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(outdir, "original_scalar_field_viridis.png"), view, ImageResolution=[1200, 900])
Hide(reader, view)

# 2) Persistence diagram + threshold by persistence (0.1) + topological simplification
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.1
threshold.UpperThreshold = 1e9  # keep pairs with persistence >= 0.1 (ThresholdMethod stays 'Between')
threshold.UpdatePipeline()

topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
topoSimplification.ScalarField = ['POINTS', 'Scalars_']
topoSimplification.UpdatePipeline()

# 3) Contour tree on simplified field
# output ports: 0 = Skeleton Nodes, 1 = Skeleton Arcs, 2 = Segmentation
contourTree = TTKContourTree(Input=topoSimplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

# arcs: tube filter for radius 1 (Tube requires vtkPolyData input)
arcsSurface = ExtractSurface(Input=OutputPort(contourTree, 1))
tube = Tube(Input=arcsSurface)
tube.Radius = 1.0
tubeDisplay = Show(tube, view)
tubeDisplay.SetRepresentationType('Surface')
tubeDisplay.SetScalarBarVisibility(view, False)
ColorBy(tubeDisplay, ('POINTS', ''))
tubeDisplay.AmbientColor = [0.7, 0.7, 0.7]
tubeDisplay.DiffuseColor = [0.7, 0.7, 0.7]

# nodes as spheres with radius 2
nodesDisplay = Show(OutputPort(contourTree, 0), view)
nodesDisplay.SetRepresentationType('Point Gaussian')
nodesDisplay.GaussianRadius = 2.0
nodesDisplay.ShaderPreset = 'Sphere'

# color nodes by CriticalType: minima blue, 1-saddles white, 2-saddles orange, maxima red
ColorBy(nodesDisplay, ('POINTS', 'CriticalType'))
critLUT = GetColorTransferFunction('CriticalType')
critLUT.InterpretValuesAsCategories = 1
critLUT.AnnotationsInitialized = 1
critLUT.Annotations = ['0', 'minimum', '1', '1-saddle', '2', '2-saddle', '3', 'maximum']
critLUT.IndexedColors = [0.0, 0.0, 1.0,   # minimum - blue
                          1.0, 1.0, 1.0,   # 1-saddle - white
                          1.0, 0.5, 0.0,   # 2-saddle - orange
                          1.0, 0.0, 0.0]   # maximum - red
nodesDisplay.SetScalarBarVisibility(view, True)

view.ResetCamera()
Render()
SaveScreenshot(os.path.join(outdir, "simplified_contour_tree.png"), view, ImageResolution=[1200, 900])

print("Done. Screenshots saved in", outdir)
