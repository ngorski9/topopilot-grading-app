import glob
import os
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

TTK_PLUGIN = '/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so'
LoadPlugin(TTK_PLUGIN, remote=False, ns=globals())

DATA_DIR = os.path.abspath('./Ionization')
files = sorted(
    glob.glob(os.path.join(DATA_DIR, 'Ionization*.vti')),
    key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))),
)

reader = XMLImageDataReader(registrationName='IonizationReader', FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

renderView = GetActiveViewOrCreate('RenderView')

# --- Raw scalar field, viridis colormap ---
rawDisplay = Show(reader, renderView, 'UniformGridRepresentation')
rawDisplay.SetRepresentationType('Volume')
ColorBy(rawDisplay, ('POINTS', 'Scalars_'))
rawDisplay.RescaleTransferFunctionToDataRange(True)
scalarsLUT = GetColorTransferFunction('Scalars_')
scalarsLUT.ApplyPreset('Viridis (matplotlib)', True)
scalarsPWF = GetOpacityTransferFunction('Scalars_')
rawDisplay.SetScalarBarVisibility(renderView, True)

# --- Persistence diagram ---
persistenceDiagram = TTKPersistenceDiagram(registrationName='PersistenceDiagram', Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

# remove the diagonal (PairIdentifier == -1)
criticalPairs = Threshold(registrationName='CriticalPairs', Input=persistenceDiagram)
criticalPairs.Scalars = ['CELLS', 'PairIdentifier']
criticalPairs.LowerThreshold = 0
criticalPairs.UpperThreshold = 999999999
criticalPairs.ThresholdMethod = 'Between'

# keep only pairs with persistence above the simplification threshold
persistentPairs = Threshold(registrationName='PersistentPairs', Input=criticalPairs)
persistentPairs.Scalars = ['CELLS', 'Persistence']
persistentPairs.LowerThreshold = 0.1
persistentPairs.UpperThreshold = 999999999
persistentPairs.ThresholdMethod = 'Between'
persistentPairs.UpdatePipeline()

# --- Topological simplification ---
simplification = TTKTopologicalSimplification(
    registrationName='TopologicalSimplification',
    Domain=reader,
    Constraints=persistentPairs,
)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplification.UpdatePipeline()

# --- Piecewise-linear contour tree on the simplified field ---
contourTree = TTKContourTree(registrationName='ContourTree', Input=simplification)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.UpdatePipeline()

nodesPort = OutputPort(contourTree, 0)
arcsPort = OutputPort(contourTree, 1)

# --- Arcs: tubes, radius 1 ---
arcsSurface = ExtractSurface(registrationName='ArcsSurface', Input=arcsPort)
tubes = Tube(registrationName='ArcTubes', Input=arcsSurface)
tubes.Radius = 1.0
tubes.NumberofSides = 12
tubesDisplay = Show(tubes, renderView, 'GeometryRepresentation')
ColorBy(tubesDisplay, None)
tubesDisplay.AmbientColor = [0.6, 0.6, 0.6]
tubesDisplay.DiffuseColor = [0.6, 0.6, 0.6]

# --- Nodes: spheres, radius 2, colored by CriticalType ---
glyph = Glyph(registrationName='NodeGlyphs', Input=nodesPort, GlyphType='Sphere')
glyph.GlyphType.Radius = 2.0
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'

glyphDisplay = Show(glyph, renderView, 'GeometryRepresentation')
ColorBy(glyphDisplay, ('POINTS', 'CriticalType'))

criticalTypeLUT = GetColorTransferFunction('CriticalType')
criticalTypeLUT.InterpretValuesAsCategories = 1
criticalTypeLUT.AnnotationsInitialized = 1
criticalTypeLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
criticalTypeLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # 0 minimum -> blue
    1.0, 1.0, 1.0,   # 1 1-saddle -> white
    1.0, 0.5, 0.0,   # 2 2-saddle -> orange
    1.0, 0.0, 0.0,   # 3 maximum -> red
]
glyphDisplay.SetScalarBarVisibility(renderView, True)

renderView.ResetCamera()
Render()

SaveScreenshot('/workspace/ionization_contour_tree.png', renderView, ImageResolution=[1600, 1200])
SaveState('/workspace/ionization_contour_tree.pvsm')
print('Done. Screenshot: /workspace/ionization_contour_tree.png, State: /workspace/ionization_contour_tree.pvsm')
