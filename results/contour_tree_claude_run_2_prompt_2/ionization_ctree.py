import os
import glob
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so', remote=True, ns=globals())

files = sorted(
    glob.glob(os.path.join('./Ionization', '*.vti')),
    key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f))))
)

reader = XMLImageDataReader(registrationName='Ionization', FileName=files)
reader.PointArrayStatus = ['Scalars_']
UpdatePipeline(time=0.0, proxy=reader)

renderView = GetActiveViewOrCreate('RenderView')
renderView.OrientationAxesVisibility = 0
renderView.ViewSize = [1600, 1200]

# ------------------------------------------------------------------
# 1) Raw scalar field, viridis colormap
# ------------------------------------------------------------------
rawDisplay = Show(reader, renderView, 'UniformGridRepresentation')
rawDisplay.SetRepresentationType('Surface')
ColorBy(rawDisplay, ('POINTS', 'Scalars_'))
rawDisplay.SetScalarBarVisibility(renderView, True)

scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)

renderView.ResetCamera()
Render(renderView)
SaveScreenshot('/workspace/ionization_raw_scalar_field.png', renderView, ImageResolution=[1600, 1200])

Hide(reader, renderView)

# ------------------------------------------------------------------
# 2) TTK pipeline: tetrahedralize -> simplify by persistence (0.1) -> contour tree
# ------------------------------------------------------------------
tetra = Tetrahedralize(registrationName='Tetrahedralize', Input=reader)

simplification = TTKTopologicalSimplificationByPersistence(
    registrationName='TopologicalSimplification', Input=tetra)
simplification.InputArray = ['POINTS', 'Scalars_']
simplification.PersistenceThreshold = 0.1
simplification.ThresholdIsAbsolute = 1
UpdatePipeline(time=0.0, proxy=simplification)

contourTree = TTKContourTree(registrationName='ContourTree', Input=simplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
UpdatePipeline(time=0.0, proxy=contourTree)

# ------------------------------------------------------------------
# Edges: tube filter, radius 1
# ------------------------------------------------------------------
ctSurface = ExtractSurface(registrationName='ContourTreeSurface', Input=contourTree)

tube = Tube(registrationName='Arcs', Input=ctSurface)
tube.Radius = 1.0
tube.Capping = 1

tubeDisplay = Show(tube, renderView, 'GeometryRepresentation')
tubeDisplay.SetRepresentationType('Surface')
ColorBy(tubeDisplay, None)
tubeDisplay.AmbientColor = [0.75, 0.75, 0.75]
tubeDisplay.DiffuseColor = [0.75, 0.75, 0.75]

# ------------------------------------------------------------------
# Vertices: sphere glyph, radius 2, colored by CriticalType
#   0 = minimum -> blue, 1 = 1-saddle -> white,
#   2 = 2-saddle -> orange, 3 = maximum -> red
# ------------------------------------------------------------------
glyph = Glyph(registrationName='CriticalPoints', Input=contourTree, GlyphType='Sphere')
glyph.GlyphType.Radius = 2.0
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'

glyphDisplay = Show(glyph, renderView, 'GeometryRepresentation')
glyphDisplay.SetRepresentationType('Surface')
ColorBy(glyphDisplay, ('POINTS', 'CriticalType'))

critLUT = GetColorTransferFunction('CriticalType')
critLUT.InterpretValuesAsCategories = 1
critLUT.AnnotationsInitialized = 1
critLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum', '4', 'Multi-Saddle']
critLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # 0 minimum -> blue
    1.0, 1.0, 1.0,   # 1 1-saddle -> white
    1.0, 0.5, 0.0,   # 2 2-saddle -> orange
    1.0, 0.0, 0.0,   # 3 maximum -> red
    0.5, 0.5, 0.5,   # 4 multi-saddle -> gray
]
glyphDisplay.SetScalarBarVisibility(renderView, True)

renderView.Background = [0.0, 0.0, 0.0]
renderView.ResetCamera()
Render(renderView)
SaveScreenshot('/workspace/ionization_contour_tree.png', renderView, ImageResolution=[1600, 1200])

print('Done. Screenshots written to /workspace/ionization_raw_scalar_field.png and /workspace/ionization_contour_tree.png')
