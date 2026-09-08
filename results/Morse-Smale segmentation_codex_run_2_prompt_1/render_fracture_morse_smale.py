from paraview.simple import *

# TTK filter definitions are supplied by ParaView's TopologyToolKit plugin.
LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so',
           remote=False, ns=globals())

data = XMLImageDataReader(FileName=['/workspace/fracture.vti'])

# Keep persistence pairs at or above 0.05, then use their endpoints as the
# topological-simplification constraints.
diagram = TTKPersistenceDiagram(Input=data)
diagram.ScalarField = ['POINTS', 'Scalars_']
significant_pairs = Threshold(Input=diagram)
significant_pairs.Scalars = ['CELLS', 'Persistence']
significant_pairs.ThresholdMethod = 'Between'
significant_pairs.LowerThreshold = 0.05
significant_pairs.UpperThreshold = 1.0e99

simplified = TTKTopologicalSimplification()
simplified.Domain = data
simplified.Constraints = significant_pairs
simplified.ScalarField = ['POINTS', 'Scalars_']

msc = TTKMorseSmaleComplex(Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.MorseSmaleComplexSegmentation = 1

# Extract output port 3: the piecewise-linear Morse-Smale segmentation.
segmentation = OutputPort(msc, 3)
critical_points = OutputPort(msc, 0)

mins = Threshold(Input=critical_points)
mins.Scalars = ['POINTS', 'CellDimension']
mins.ThresholdMethod = 'Between'
mins.LowerThreshold = 0
mins.UpperThreshold = 0

saddles = Threshold(Input=critical_points)
saddles.Scalars = ['POINTS', 'CellDimension']
saddles.ThresholdMethod = 'Between'
saddles.LowerThreshold = 1
saddles.UpperThreshold = 1

maxes = Threshold(Input=critical_points)
maxes.Scalars = ['POINTS', 'CellDimension']
maxes.ThresholdMethod = 'Between'
maxes.LowerThreshold = 2
maxes.UpperThreshold = 2

def sphere_glyph(source):
    glyph = Glyph(Input=source, GlyphType='Sphere')
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1
    glyph.GlyphType.Radius = 2.0
    glyph.GlyphType.ThetaResolution = 20
    glyph.GlyphType.PhiResolution = 20
    return glyph

min_glyphs = sphere_glyph(mins)
saddle_glyphs = sphere_glyph(saddles)
max_glyphs = sphere_glyph(maxes)

view = CreateView('RenderView')
view.ViewSize = [1400, 760]
view.Background = [0.08, 0.08, 0.10]
view.BackgroundColorMode = 'Single Color'
view.InteractionMode = '2D'

seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.Representation = 'Surface'
seg_display.SetScalarBarVisibility(view, False)
seg_display.LookupTable.ColorSpace = 'HSV'
seg_display.LookupTable.HSVWrap = 1
seg_display.LookupTable.NumberOfTableValues = 256
seg_display.LookupTable.RGBPoints = [0.0, 0.12, 0.28, 0.72,
                                     0.5, 0.20, 0.72, 0.35,
                                     1.0, 0.82, 0.26, 0.12]

for glyph, color in ((min_glyphs, [0.0, 0.25, 1.0]),
                     (saddle_glyphs, [1.0, 1.0, 1.0]),
                     (max_glyphs, [1.0, 0.0, 0.0])):
    display = Show(glyph, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.35
    display.Specular = 0.25

ResetCamera(view)
view.CameraParallelProjection = 1
ResetCamera(view)
Render(view)
SaveScreenshot('/workspace/fracture_morse_smale.png', view, ImageResolution=[1400, 760])
SaveState('/workspace/fracture_morse_smale.pvsm')
