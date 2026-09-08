from paraview.simple import *

# Input and persistence simplification (relative threshold, as used by TTK).
source = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
simplified = TTKTopologicalSimplificationByPersistence(Input=source)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 0

# Piecewise-linear Morse--Smale complex of the simplified field.
msc = TTKMorseSmaleComplex(Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.MorseSmaleComplexSegmentation = 1
msc.CriticalPoints = 1
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.UpdatePipeline()

# Port 3 is TTK's piecewise-linear Morse--Smale segmentation; port 0 holds
# critical points, classified by CellDimension (0=min, 1=saddle, 2=max).
segmentation = OutputPort(msc, 3)
critical_points = OutputPort(msc, 0)

view = CreateView('RenderView')
view.ViewSize = [1600, 850]
view.Background = [0.08, 0.08, 0.08]
view.OrientationAxesVisibility = 0
view.InteractionMode = '2D'

seg_display = Show(segmentation, view)
seg_display.Representation = 'Surface'
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
lut = GetColorTransferFunction('MorseSmaleManifold')
lut.InterpretValuesAsCategories = 0
lut.RGBPoints = [0.0, 0.18, 0.30, 0.75, 8.0, 0.10, 0.75, 0.70,
                 16.0, 0.93, 0.75, 0.12, 24.0, 0.85, 0.22, 0.48,
                 31.0, 0.35, 0.12, 0.68]
lut.NumberOfTableValues = 32
lut.Discretize = 1
seg_display.RescaleTransferFunctionToDataRange(True, False)
seg_display.SetScalarBarVisibility(view, False)

def points_of_dimension(dimension, rgb, label):
    selected = Threshold(Input=critical_points)
    selected.Scalars = ['POINTS', 'CellDimension']
    selected.ThresholdMethod = 'Between'
    selected.LowerThreshold = dimension
    selected.UpperThreshold = dimension
    glyph = Glyph(Input=selected, GlyphType='Sphere')
    glyph.GlyphType.Radius = 2.0
    glyph.ScaleArray = ['POINTS', '']
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = 'All Points'
    display = Show(glyph, view)
    display.DiffuseColor = rgb
    display.AmbientColor = rgb
    display.Ambient = 0.35
    display.Specular = 0.2
    display.Representation = 'Surface'
    display.SetScalarBarVisibility(view, False)
    RenameSource(label, glyph)
    return glyph

minimums = points_of_dimension(0, [0.05, 0.25, 1.0], 'Minimums (radius 2, blue)')
saddles = points_of_dimension(1, [1.0, 1.0, 1.0], 'Saddles (radius 2, white)')
maximums = points_of_dimension(2, [1.0, 0.05, 0.05], 'Maximums (radius 2, red)')

view.ResetCamera()
view.CameraParallelProjection = 1
view.ResetCamera()
view.CameraPosition = [124.0, 60.0, 500.0]
view.CameraFocalPoint = [124.0, 60.0, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelScale = 140.0

Render(view)
SaveScreenshot('/workspace/fracture_morse_smale.png', view, ImageResolution=[1600, 850])
SaveState('/workspace/fracture_morse_smale.pvsm')
