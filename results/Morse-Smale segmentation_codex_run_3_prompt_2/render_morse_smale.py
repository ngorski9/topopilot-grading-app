from paraview.simple import *

# Simplify the scalar field, then compute its piecewise-linear Morse-Smale complex.
reader = XMLImageDataReader(registrationName='fracture.vti', FileName=['/workspace/fracture.vti'])
reader.PointArrayStatus = ['Scalars_']

simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.05)', Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 0

msc = TTKMorseSmaleComplex(registrationName='Morse-Smale complex', Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.MorseSmaleComplexSegmentation = 1
msc.CriticalPoints = 1
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.Ascending1Separatrices = 1
msc.Descending1Separatrices = 1

view = CreateView('RenderView')
view.ViewSize = [1600, 900]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0
view.InteractionMode = '2D'

# Output port 3 is the piecewise-linear Morse-Smale segmentation.
segmentation = OutputPort(msc, 3)
seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.Representation = 'Surface'
seg_display.InterpolateScalarsBeforeMapping = 0
seg_display.SetScalarBarVisibility(view, False)
seg_lut = GetColorTransferFunction('MorseSmaleManifold')
seg_lut.InterpretValuesAsCategories = 1
seg_lut.Annotations = [str(i) for i in range(32) for _ in (0, 1)]
for i in range(32):
    seg_lut.Annotations[2 * i] = str(i)
    seg_lut.Annotations[2 * i + 1] = 'Region ' + str(i)
seg_lut.IndexedColors = [
    0.23, 0.47, 0.70, 0.89, 0.35, 0.20, 0.29, 0.69, 0.36, 0.95, 0.66, 0.20,
    0.55, 0.34, 0.69, 0.47, 0.72, 0.70, 0.82, 0.46, 0.63, 0.30, 0.30, 0.30,
]

# Output port 0 contains the critical points.  CellDimension identifies
# minima (0), saddles (1), and maxima (2) for this 2D domain.
critical_points = OutputPort(msc, 0)
def critical_layer(name, dimension, color):
    selected = Threshold(registrationName=name + ' selection', Input=critical_points)
    selected.Scalars = ['POINTS', 'CellDimension']
    selected.ThresholdMethod = 'Between'
    selected.LowerThreshold = dimension
    selected.UpperThreshold = dimension
    glyph = Glyph(registrationName=name, Input=selected, GlyphType='Sphere')
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphType.Radius = 2.0
    glyph.GlyphType.ThetaResolution = 24
    glyph.GlyphType.PhiResolution = 24
    display = Show(glyph, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.35
    display.Specular = 0.25
    display.SpecularPower = 20.0
    return glyph

minimums = critical_layer('Minimums (blue)', 0, [0.10, 0.35, 1.00])
saddles = critical_layer('Saddles (white)', 1, [1.00, 1.00, 1.00])
maximums = critical_layer('Maximums (red)', 2, [1.00, 0.08, 0.08])

view.ResetCamera()
view.CameraParallelProjection = 1
view.ResetCamera()
Render(view)
SaveScreenshot('/workspace/fracture_morse_smale.png', view, ImageResolution=[1600, 900])
SaveState('/workspace/fracture_morse_smale.pvsm')
