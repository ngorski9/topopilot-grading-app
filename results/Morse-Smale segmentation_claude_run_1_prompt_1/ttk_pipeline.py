from paraview.simple import *
import paraview

paraview.simple._DisableFirstRenderCameraReset()

SCALAR = "Scalars_"
PERSISTENCE_THRESHOLD = 0.05

# 1. Load data
reader = OpenDataFile("fracture.vti")
reader.UpdatePipeline()

# 2. Compute persistence diagram of the input field
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', SCALAR]
persistenceDiagram.UpdatePipeline()

# 3. Keep only pairs with persistence above the threshold (skip the last
#    "global" pair which always has infinite/max persistence and must stay)
criticalPairs = Threshold(Input=persistenceDiagram)
criticalPairs.Scalars = ['CELLS', 'Persistence']
criticalPairs.ThresholdMethod = 'Above Upper Threshold'
criticalPairs.UpperThreshold = PERSISTENCE_THRESHOLD
criticalPairs.UpdatePipeline()

# 4. Simplify the scalar field so that only the surviving critical points remain
topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=criticalPairs)
topoSimplification.ScalarField = ['POINTS', SCALAR]
topoSimplification.UpdatePipeline()

# 5. Compute the piecewise-linear Morse-Smale complex on the simplified field
msc = TTKMorseSmaleComplex(Input=topoSimplification)
msc.ScalarField = ['POINTS', SCALAR]
msc.CriticalPoints = 1
msc.Ascending1Separatrices = 1
msc.Descending1Separatrices = 1
msc.SaddleConnectors = 1
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.MorseSmaleComplexSegmentation = 1
msc.UpdatePipeline()

# msc has 4 outputs: 0=Critical Points, 1=1-Separatrices, 2=2-Separatrices, 3=Segmentation
criticalPoints = OutputPort(msc, 0)
segmentation = OutputPort(msc, 3)

renderView = GetActiveViewOrCreate('RenderView')

# 6. Show the Morse-Smale segmentation, colored by the SegmentationId/region
segDisplay = Show(segmentation, renderView)
segDisplay.Representation = 'Surface'
ColorBy(segDisplay, ('POINTS', 'DescendingManifold'))
segDisplay.SetScalarBarVisibility(renderView, False)

# 7. Split critical points by type and glyph them as spheres.
#    On this 2D field (z-extent 0), TTK reports the type via CellDimension:
#    0 = minimum, 1 = saddle, 2 = maximum
mins = Threshold(Input=criticalPoints)
mins.Scalars = ['POINTS', 'CellDimension']
mins.ThresholdMethod = 'Between'
mins.LowerThreshold = 0
mins.UpperThreshold = 0

maxs = Threshold(Input=criticalPoints)
maxs.Scalars = ['POINTS', 'CellDimension']
maxs.ThresholdMethod = 'Between'
maxs.LowerThreshold = 2
maxs.UpperThreshold = 2

saddles = Threshold(Input=criticalPoints)
saddles.Scalars = ['POINTS', 'CellDimension']
saddles.ThresholdMethod = 'Between'
saddles.LowerThreshold = 1
saddles.UpperThreshold = 1

def show_glyph(src, color, name):
    g = Glyph(Input=src, GlyphType='Sphere')
    g.GlyphType.Radius = 2.0
    g.ScaleFactor = 1.0
    g.GlyphMode = 'All Points'
    RenameSource(name, g)
    d = Show(g, renderView)
    d.Representation = 'Surface'
    d.ColorArrayName = [None, '']
    d.AmbientColor = color
    d.DiffuseColor = color
    return g, d

show_glyph(maxs, [1.0, 0.0, 0.0], 'Maxima')
show_glyph(saddles, [1.0, 1.0, 1.0], 'Saddles')
show_glyph(mins, [0.0, 0.0, 1.0], 'Minima')

renderView.Background = [0.2, 0.2, 0.2]
renderView.ResetCamera()
renderView.OrientationAxesVisibility = 0

Render()
SaveScreenshot('/workspace/morse_smale_result.png', renderView, ImageResolution=[1600, 1200])
SaveState('/workspace/morse_smale_result.pvsm')
print("Done. Screenshot: /workspace/morse_smale_result.png, State: /workspace/morse_smale_result.pvsm")
