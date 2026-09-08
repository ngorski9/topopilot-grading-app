from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

SCALAR = 'Scalars_'
PERSISTENCE_THRESHOLD = 0.05

# 1. Load data
reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.UpdatePipeline()

# 2. Compute persistence diagram
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', SCALAR]
persistenceDiagram.UpdatePipeline()

# 3. Keep only pairs with persistence >= threshold (drop the diagonal, which has Persistence == 0)
criticalPairs = Threshold(Input=persistenceDiagram)
criticalPairs.Scalars = ['CELLS', 'Persistence']
criticalPairs.LowerThreshold = PERSISTENCE_THRESHOLD
criticalPairs.UpperThreshold = 1e9
criticalPairs.ThresholdMethod = 'Between'
criticalPairs.UpdatePipeline()

# 4. Simplify the scalar field topologically using the surviving persistence pairs as constraints
topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=criticalPairs)
topoSimplification.ScalarField = ['POINTS', SCALAR]
topoSimplification.VertexIdentifierField = ['POINTS', 'CriticalPointsIdentifier']
topoSimplification.UpdatePipeline()

# 5. Compute the (piecewise-linear) Morse-Smale complex on the simplified field
morseSmale = TTKMorseSmaleComplex(Input=topoSimplification)
morseSmale.ScalarField = ['POINTS', SCALAR]
morseSmale.CriticalPoints = 1
morseSmale.Ascending1Separatrices = 1
morseSmale.Descending1Separatrices = 1
morseSmale.SaddleConnectors = 1
morseSmale.AscendingSegmentation = 1
morseSmale.DescendingSegmentation = 1
morseSmale.MorseSmaleComplexSegmentation = 1
morseSmale.UpdatePipeline()

segmentation = OutputPort(morseSmale, 3)
criticalPoints = OutputPort(morseSmale, 0)

renderView = GetActiveViewOrCreate('RenderView')

# 6. Show the Morse-Smale segmentation
segDisplay = Show(segmentation, renderView)
segDisplay.Representation = 'Surface'
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segDisplay.SetScalarBarVisibility(renderView, False)
segDisplay.RescaleTransferFunctionToDataRange(True)

# 7. Split critical points by type using CellDimension:
#    0 = local minimum, 1 = saddle (2D), 2 = local maximum (2D)
mins = Threshold(Input=criticalPoints)
mins.Scalars = ['POINTS', 'CellDimension']
mins.LowerThreshold = 0
mins.UpperThreshold = 0
mins.ThresholdMethod = 'Between'

maxs = Threshold(Input=criticalPoints)
maxs.Scalars = ['POINTS', 'CellDimension']
maxs.LowerThreshold = 2
maxs.UpperThreshold = 2
maxs.ThresholdMethod = 'Between'

saddles = Threshold(Input=criticalPoints)
saddles.Scalars = ['POINTS', 'CellDimension']
saddles.LowerThreshold = 1
saddles.UpperThreshold = 1
saddles.ThresholdMethod = 'Between'

def show_glyph(src, color, radius=2.0):
    glyph = Glyph(Input=src, GlyphType='Sphere')
    glyph.GlyphType.Radius = radius
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = 'All Points'
    disp = Show(glyph, renderView)
    disp.DiffuseColor = color
    disp.AmbientColor = color
    return glyph, disp

minsGlyph, minsDisp = show_glyph(mins, [0.0, 0.0, 1.0])
maxsGlyph, maxsDisp = show_glyph(maxs, [1.0, 0.0, 0.0])
saddlesGlyph, saddlesDisp = show_glyph(saddles, [1.0, 1.0, 1.0])

renderView.ResetCamera()
Render()

SaveScreenshot('/workspace/morse_smale_segmentation.png', renderView, ImageResolution=[1600, 900])
print('done')
