from paraview.simple import *
import paraview

paraview.simple._DisableFirstRenderCameraReset()

LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so', remote=False, ns=globals())

reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.PointArrayStatus = ['Scalars_']

# TTK filters require the array to be flagged active scalars
tetra = TTKPersistenceDiagram(Input=reader)
tetra.ScalarField = ['POINTS', 'Scalars_']
tetra.InputOffsetField = ['POINTS', 'Scalars_']
tetra.UpdatePipeline()

# Simplify the persistence diagram at threshold 0.05 (persistence value, not %)
critPairs = Threshold(Input=tetra)
critPairs.Scalars = ['CELLS', 'Persistence']
critPairs.LowerThreshold = 0.05
critPairs.UpperThreshold = 1e9
critPairs.ThresholdMethod = 'Above Upper Threshold'
critPairs.UpdatePipeline()

simplify = TTKTopologicalSimplification(Domain=reader, Constraints=critPairs)
simplify.ScalarField = ['POINTS', 'Scalars_']
simplify.InputOffsetField = ['POINTS', 'Scalars_']
simplify.VertexIdentifierField = ['POINTS', 'CriticalPoints_PointId']
simplify.UpdatePipeline()

morse = TTKMorseSmaleComplex(Input=simplify)
morse.ScalarField = ['POINTS', 'Scalars_']
morse.OffsetField = ['POINTS', 'Scalars_']
morse.CriticalPoints = 1
morse.AscendingSegmentation = 1
morse.DescendingSegmentation = 1
morse.MorseSmaleComplexSegmentation = 1
morse.SaddleConnectors = 0
morse.Ascending1Separatrices = 0
morse.Descending1Separatrices = 0
morse.Ascending2Separatrices = 0
morse.Descending2Separatrices = 0
morse.UpdatePipeline()

segmentation = OutputPort(morse, 3)
critPoints = OutputPort(morse, 0)

renderView = GetActiveViewOrCreate('RenderView')

segDisplay = Show(segmentation, renderView)
segDisplay.Representation = 'Surface'
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segDisplay.RescaleTransferFunctionToDataRange(True)
segLut = GetColorTransferFunction('MorseSmaleManifold')
segLut.ApplyPreset('Rainbow Uniform', True)
segLut.InterpretValuesAsCategories = 0

glyph = Glyph(Input=critPoints, GlyphType='Sphere')
glyph.GlyphType = 'Sphere'
glyph.GlyphType.Radius = 2.0
glyph.ScaleFactor = 1.0
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.GlyphMode = 'All Points'
Hide(critPoints, renderView)

glyphDisplay = Show(glyph, renderView)
glyphDisplay.Representation = 'Surface'

# TTK critical point dimension encoding for 2D scalar field:
# 0 = minimum, 1 = 1-saddle, 2 = maximum
ColorBy(glyphDisplay, ('POINTS', 'CellDimension'))
lut = GetColorTransferFunction('CellDimension')
lut.InterpretValuesAsCategories = 1
lut.AnnotationsInitialized = 1
lut.Annotations = ['0', 'Minimum', '1', 'Saddle', '2', 'Maximum']
lut.IndexedColors = [
    0.0, 0.0, 1.0,   # Minimum -> blue
    1.0, 1.0, 1.0,   # Saddle -> white
    1.0, 0.0, 0.0,   # Maximum -> red
]

renderView.ResetCamera()
renderView.Update()
Render(renderView)

SaveScreenshot('/workspace/morse_smale_render.png', renderView, ImageResolution=[1600, 900])
print("Saved render to /workspace/morse_smale_render.png")
