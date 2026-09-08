from paraview.simple import *
import glob, os

paraview.simple._DisableFirstRenderCameraReset()

files = sorted(glob.glob('/workspace/Ionization/Ionization*.vti'),
               key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
UpdatePipeline(time=0, proxy=reader)

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.ViewSize = [1400, 1000]

# --- Raw scalar field with viridis ---
rawDisplay = Show(reader, renderView1)
ColorBy(rawDisplay, ('POINTS', 'Scalars_'))
rawDisplay.SetScalarBarVisibility(renderView1, True)
scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)
rawDisplay.SetRepresentationType('Surface')

renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/raw_scalar_field.png', renderView1)

Hide(reader, renderView1)

# --- Persistence Diagram ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
UpdatePipeline(time=0, proxy=persistenceDiagram)

# --- Threshold on persistence pairs (Persistence > 0.1) ---
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.ThresholdMethod = 'Above Upper Threshold'
threshold.UpperThreshold = 0.1

UpdatePipeline(time=0, proxy=threshold)

# --- Topological Simplification using the thresholded persistence pairs ---
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
UpdatePipeline(time=0, proxy=simplification)

# --- Contour Tree on the simplified field ---
contourTree = TTKContourTree(Input=simplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
UpdatePipeline(time=0, proxy=contourTree)

# The output is a single unstructured grid: points = nodes, line cells = arcs
nodes = contourTree
arcs = contourTree

# --- Edges as tubes ---
arcsGeom = ExtractSurface(Input=arcs)
tube = Tube(Input=arcsGeom)
tube.Radius = 1.0
tubeDisplay = Show(tube, renderView1)
tubeDisplay.SetRepresentationType('Surface')
tubeDisplay.ColorArrayName = [None, '']
tubeDisplay.AmbientColor = [0.7, 0.7, 0.7]
tubeDisplay.DiffuseColor = [0.7, 0.7, 0.7]

# --- Vertices as spheres, colored by CriticalType ---
sphere = Glyph(Input=nodes, GlyphType='Sphere')
sphere.GlyphType.Radius = 2.0
sphere.ScaleArray = ['POINTS', 'No scale array']
sphere.ScaleFactor = 1.0
sphere.GlyphMode = 'All Points'
sphereDisplay = Show(sphere, renderView1)
sphereDisplay.SetRepresentationType('Surface')
ColorBy(sphereDisplay, ('POINTS', 'CriticalType'))

critLUT = GetColorTransferFunction('CriticalType')
critLUT.InterpretValuesAsCategories = 1
critLUT.AnnotationsInitialized = 1
# CriticalType values: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum
critLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
critLUT.IndexedColors = [0.0, 0.0, 1.0,   # minimum -> blue
                          1.0, 1.0, 1.0,  # 1-saddle -> white
                          1.0, 0.5, 0.0,  # 2-saddle -> orange
                          1.0, 0.0, 0.0]  # maximum -> red
sphereDisplay.SetScalarBarVisibility(renderView1, True)

renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/contour_tree_simplified.png', renderView1)

print("Done.")
