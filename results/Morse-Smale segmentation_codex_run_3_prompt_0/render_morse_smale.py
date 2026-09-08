from paraview.simple import *

# Read and simplify the scalar field with a normalized persistence threshold.
source = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
simplified = TTKTopologicalSimplificationByPersistence(Input=source)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 0

# Compute the piecewise-linear Morse--Smale complex of the simplified field.
msc = TTKMorseSmaleComplex(Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.MorseSmaleComplexSegmentation = 1
msc.CriticalPoints = 1
msc.Ascending1Separatrices = 1
msc.Descending1Separatrices = 1

# Output port 3 is the PL Morse--Smale segmentation; port 0 contains critical points.
segmentation = OutputPort(msc, 3)
critical_points = OutputPort(msc, 0)

def select_dimension(value):
    selected = Threshold(Input=critical_points)
    selected.Scalars = ['POINTS', 'CellDimension']
    selected.ThresholdMethod = 'Between'
    selected.LowerThreshold = value
    selected.UpperThreshold = value
    return selected

def critical_glyph(selected):
    glyph = Glyph(Input=selected, GlyphType='Sphere')
    glyph.GlyphMode = 'All Points'
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphType.Radius = 2.0
    glyph.GlyphType.ThetaResolution = 20
    glyph.GlyphType.PhiResolution = 20
    return glyph

minima = critical_glyph(select_dimension(0))
saddles = critical_glyph(select_dimension(1))
maxima = critical_glyph(select_dimension(2))

view = CreateView('RenderView')
view.ViewSize = [1400, 1000]
view.Background = [0.08, 0.08, 0.10]
view.BackgroundColorMode = 'Gradient'
view.Background2 = [0.22, 0.22, 0.27]

seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.Representation = 'Surface'
seg_display.Opacity = 0.82
seg_display.SetScalarBarVisibility(view, True)
GetColorTransferFunction('MorseSmaleManifold').ApplyPreset('Turbo', True)

for proxy, color in ((minima, [0.0, 0.25, 1.0]),
                     (saddles, [1.0, 1.0, 1.0]),
                     (maxima, [1.0, 0.0, 0.0])):
    display = Show(proxy, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Specular = 0.35
    display.SpecularPower = 25.0

view.CameraParallelProjection = 1
view.ResetCamera()
view.CameraParallelScale *= 1.12
Render(view)
SaveScreenshot('/workspace/fracture_morse_smale.png', view, ImageResolution=[1400, 1000])
SaveState('/workspace/fracture_morse_smale.pvsm')
