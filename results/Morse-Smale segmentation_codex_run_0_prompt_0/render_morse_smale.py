from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

reader = XMLImageDataReader(registrationName='fracture.vti', FileName=['/workspace/fracture.vti'])

# Cancel all persistence pairs below the requested absolute threshold.
simplified = TTKTopologicalSimplificationByPersistence(registrationName='Persistence simplification (0.05)', Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 1

# Compute the PL Morse--Smale complex from the simplified field.
msc = TTKMorseSmaleComplex(registrationName='PL Morse-Smale complex', Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.ThresholdIsAbsolute = 1
msc.MorseSmaleComplexSegmentation = 1
msc.CriticalPoints = 1
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.Ascending1Separatrices = 1
msc.Descending1Separatrices = 1

# Output 3 is the image-data piecewise-linear Morse--Smale segmentation.
segmentation = OutputPort(msc, 3)

view = CreateView('RenderView')
view.ViewSize = [1600, 1100]
view.Background = [0.08, 0.09, 0.12]
view.BackgroundColorMode = 'Gradient'
view.Background2 = [0.18, 0.20, 0.27]

seg_display = Show(segmentation, view)
seg_display.Representation = 'Surface'
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.RescaleTransferFunctionToDataRange(True, False)
seg_display.SetScalarBarVisibility(view, True)
seg_display.Opacity = 0.72

# Extract and render every critical-point class separately.  TTK encodes
# The input is two-dimensional, so TTK encodes minima=0, saddles=1, maxima=2.
critical = OutputPort(msc, 0)
def critical_glyph(name, value, color):
    selected = Threshold(registrationName=name, Input=critical)
    selected.Scalars = ['POINTS', 'CellDimension']
    selected.LowerThreshold = value
    selected.UpperThreshold = value
    selected.ThresholdMethod = 'Between'
    glyph = Glyph(registrationName=name + ' (radius 2)', Input=selected, GlyphType='Sphere')
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.GlyphMode = 'All Points'
    glyph.ScaleFactor = 2.0
    glyph.GlyphType.Radius = 1.0
    glyph.GlyphType.ThetaResolution = 20
    glyph.GlyphType.PhiResolution = 20
    display = Show(glyph, view)
    display.ColorArrayName = [None, '']
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.35
    display.Specular = 0.3
    return glyph

minima = critical_glyph('Minima', 0, [0.10, 0.35, 1.00])
saddles = critical_glyph('Saddles', 1, [1.0, 1.0, 1.0])
maxima = critical_glyph('Maxima', 2, [1.0, 0.05, 0.05])

view.CameraPosition = [124, 60, 500]
view.CameraFocalPoint = [124, 60, 0]
view.CameraViewUp = [0, 1, 0]
view.CameraParallelProjection = 1
view.CameraParallelScale = 150
Render(view)
SaveScreenshot('/workspace/fracture_morse_smale.png', view, ImageResolution=[1600, 1100])
SaveState('/workspace/fracture_morse_smale.pvsm')
