from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

# --- Load TTK plugin ---
LoadDistributedPlugin('TopologyToolKit', ns=globals())

# --- Load the time-varying Ionization dataset ---
data_dir = '/workspace/Ionization'
files = sorted(
    [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.vti')],
    key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))))
)

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

renderView1 = GetActiveViewOrCreate('RenderView')

# --- Visualize original scalar field with viridis colormap ---
scalarDisplay = Show(reader, renderView1)
scalarDisplay.SetRepresentationType('Volume')
ColorBy(scalarDisplay, ('POINTS', 'Scalars_'))
scalarDisplay.SetScalarBarVisibility(renderView1, True)
scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)

renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/scalar_field_viridis.png', renderView1, ImageResolution=[1280, 800])

Hide(reader, renderView1)

# --- Persistence Diagram (required to build simplification threshold) ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

# --- Threshold the persistence diagram at 0.1 ---
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.1
threshold.UpperThreshold = 1e9
threshold.ThresholdMethod = 'Between'
threshold.UpdatePipeline()

# --- Topological Simplification using the persistence threshold ---
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplification.UpdatePipeline()

# --- Contour Tree (piecewise-linear) on the simplified field ---
contourTree = TTKContourTree(Input=simplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

# TTKContourTree outputs: Port 0 = nodes, Port 1 = arcs, Port 2 = segmentation
nodesOutput = OutputPort(contourTree, 0)
arcsOutput = OutputPort(contourTree, 1)

# --- Display arcs (tree edges) as tubes ---
arcsPoly = ExtractSurface(Input=arcsOutput)
tubes = Tube(Input=arcsPoly)
tubes.Radius = 1.0
tubesDisplay = Show(tubes, renderView1)
tubesDisplay.DiffuseColor = [0.8, 0.8, 0.8]

# --- Display nodes (tree vertices), colored by CriticalType ---
spheres = Glyph(Input=nodesOutput, GlyphType='Sphere')
spheres.GlyphType.Radius = 2.0
spheres.ScaleFactor = 1.0
spheres.GlyphMode = 'All Points'

spheresDisplay = Show(spheres, renderView1)
ColorBy(spheresDisplay, ('POINTS', 'CriticalType'))

criticalTypeLUT = GetColorTransferFunction('CriticalType')
criticalTypeLUT.InterpretValuesAsCategories = 1
criticalTypeLUT.AnnotationsInitialized = 1

# TTK CriticalType encoding: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum
criticalTypeLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
criticalTypeLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # Minimum -> blue
    1.0, 1.0, 1.0,   # 1-Saddle -> white
    1.0, 0.5, 0.0,   # 2-Saddle -> orange
    1.0, 0.0, 0.0,   # Maximum -> red
]

spheresDisplay.SetScalarBarVisibility(renderView1, True)

renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/contour_tree_simplified.png', renderView1, ImageResolution=[1280, 800])

print('Done. Screenshots saved to /workspace/scalar_field_viridis.png and /workspace/contour_tree_simplified.png')
