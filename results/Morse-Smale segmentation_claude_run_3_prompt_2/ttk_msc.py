from paraview.simple import *

LoadDistributedPlugin('TopologyToolKit', ns=globals())
paraview.simple._DisableFirstRenderCameraReset()

reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.PointArrayStatus = ['Scalars_']

# 1) Persistence diagram of the raw field
diagram = TTKPersistenceDiagram(Input=reader)
diagram.ScalarField = ['POINTS', 'Scalars_']

# 2) Keep only persistence pairs above the 0.05 threshold
threshold = Threshold(Input=diagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.ThresholdMethod = 'Above Upper Threshold'
threshold.UpperThreshold = 0.05

# 3) Simplify the scalar field so that only those pairs remain
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']

# 4) Piecewise-linear Morse-Smale complex of the simplified field
msc = TTKMorseSmaleComplex(Input=simplification)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.CriticalPoints = 1
msc.Ascending1Separatrices = 0
msc.Descending1Separatrices = 0
msc.Ascending2Separatrices = 0
msc.Descending2Separatrices = 0
msc.SaddleConnectors = 0
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.MorseSmaleComplexSegmentation = 1
msc.UpdatePipeline()

critPoints = OutputPort(msc, 0)     # critical points; CellDimension: 0=min,1=saddle,2=max
segmentation = OutputPort(msc, 3)   # final MS segmentation (MorseSmaleManifold)

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]
view.OrientationAxesVisibility = 0

# --- Segmentation surface ---
segDisplay = Show(segmentation, view)
segDisplay.Representation = 'Surface'
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segLUT = GetColorTransferFunction('MorseSmaleManifold')
segLUT.ApplyPreset('Cool to Warm', True)
segDisplay.SetScalarBarVisibility(view, False)

# --- Critical points as spheres, colored by type via CellDimension ---
glyph = Glyph(Input=critPoints, GlyphType='Sphere')
glyph.GlyphType.Radius = 2.0
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'

glyphDisplay = Show(glyph, view)
glyphDisplay.Representation = 'Surface'
ColorBy(glyphDisplay, ('POINTS', 'CellDimension'))

lut = GetColorTransferFunction('CellDimension')
lut.InterpretValuesAsCategories = 1
lut.AnnotationsInitialized = 1
# 0 = minimum -> blue, 1 = saddle -> white, 2 = maximum -> red
lut.Annotations = ['0', 'Minimum', '1', 'Saddle', '2', 'Maximum']
lut.IndexedColors = [
    0.0, 0.0, 1.0,   # minimum: blue
    1.0, 1.0, 1.0,   # saddle: white
    1.0, 0.0, 0.0,   # maximum: red
]
glyphDisplay.SetScalarBarVisibility(view, False)

view.Background = [0.2, 0.2, 0.2]
ResetCamera(view)
view.CameraParallelProjection = 1

Render(view)
SaveScreenshot('/workspace/fracture_msc.png', view, ImageResolution=[1600, 1200])
print('Saved screenshot to /workspace/fracture_msc.png')
