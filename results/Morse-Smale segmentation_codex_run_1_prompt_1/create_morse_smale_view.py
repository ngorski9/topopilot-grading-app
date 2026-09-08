from paraview.simple import *

# Load the scalar field and simplify it by absolute persistence.
field = XMLImageDataReader(registrationName='fracture.vti', FileName=['/workspace/fracture.vti'])
field.PointArrayStatus = ['Scalars_']
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.05)', Input=field)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.05
simplified.ThresholdIsAbsolute = 1

# TTK's PL Morse--Smale complex. Output port 3 is the segmentation and port 0
# contains its critical points.
msc = TTKMorseSmaleComplex(registrationName='PL Morse-Smale complex', Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.ThresholdIsAbsolute = 1
msc.MorseSmaleComplexSegmentation = 1
msc.CriticalPoints = 1

critical = OutputPort(msc, 0)
segmentation = OutputPort(msc, 3)

def critical_type(name, dimension, color):
    selected = Threshold(registrationName=name, Input=critical)
    selected.Scalars = ['POINTS', 'CellDimension']
    selected.ThresholdMethod = 'Between'
    selected.LowerThreshold = dimension
    selected.UpperThreshold = dimension
    glyph = Glyph(registrationName=name + ' (radius 2)', Input=selected,
                  GlyphType='Sphere')
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphType.Radius = 2.0
    return glyph, color

mins, blue = critical_type('Minima', 0, [0.0, 0.25, 1.0])
saddles, white = critical_type('Saddles', 1, [1.0, 1.0, 1.0])
maxes, red = critical_type('Maxima', 2, [1.0, 0.0, 0.0])

view = CreateView('RenderView')
view.ViewSize = [1400, 800]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0

seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.Opacity = 0.72
seg_display.Representation = 'Surface'
seg_display.SetScalarBarVisibility(view, True)
GetColorTransferFunction('MorseSmaleManifold').ApplyPreset('Rainbow Uniform', True)

for source, color in [(mins, blue), (saddles, white), (maxes, red)]:
    display = Show(source, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.35
    display.Specular = 0.15

view.ResetCamera()
view.CameraParallelProjection = 1
view.CameraParallelScale = 75
Render(view)
SaveScreenshot('/workspace/fracture_morse_smale.png', view, ImageResolution=[1400, 800])
SaveState('/workspace/fracture_morse_smale.pvsm')
