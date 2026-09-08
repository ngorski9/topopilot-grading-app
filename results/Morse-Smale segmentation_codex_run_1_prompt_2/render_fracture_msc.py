from paraview.simple import *

# Persistence simplification followed by the piecewise-linear Morse--Smale complex.
source = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
simplified = TTKTopologicalSimplificationByPersistence(Input=source)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 0

msc = TTKMorseSmaleComplex(Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.CriticalPoints = 1
msc.MorseSmaleComplexSegmentation = 1

# TTK output 0 = critical points; output 3 = piecewise-linear segmentation.
critical = OutputPort(msc, 0)
segmentation = OutputPort(msc, 3)

view = CreateView('RenderView')
view.ViewSize = [1400, 1000]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0

seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'Scalars_'))
seg_display.Representation = 'Surface'
seg_display.Opacity = 0.82
seg_display.InterpolateScalarsBeforeMapping = 0
seg_display.SetScalarBarVisibility(view, True)
lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Cool to Warm', True)

def critical_markers(lo, hi, color):
    selected = Threshold(Input=critical)
    selected.Scalars = ['POINTS', 'CellDimension']
    selected.LowerThreshold = lo
    selected.UpperThreshold = hi
    selected.ThresholdMethod = 'Between'
    marker = Glyph(Input=selected, GlyphType='Sphere')
    marker.GlyphType.Radius = 2.0
    marker.GlyphType.ThetaResolution = 20
    marker.GlyphType.PhiResolution = 20
    marker.ScaleArray = ['POINTS', '']
    marker.ScaleFactor = 1.0
    display = Show(marker, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.ColorArrayName = [None, '']
    return marker

# CellDimension 0, 1, and 2 correspond to minima, saddles, and maxima.
minima = critical_markers(0, 0, [0.05, 0.20, 1.0])
saddles = critical_markers(1, 1, [1.0, 1.0, 1.0])
maxima = critical_markers(2, 2, [1.0, 0.05, 0.05])

Render(view)
view.ResetCamera()
view.CameraParallelProjection = 1
Render(view)
SaveScreenshot('/workspace/fracture_morse_smale.png', view, ImageResolution=[1400, 1000])
SaveState('/workspace/fracture_morse_smale.pvsm')
