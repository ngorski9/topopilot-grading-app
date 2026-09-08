from paraview.simple import *

# Input and persistence-based topological simplification (threshold = 0.05).
reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.PointArrayStatus = ['Scalars_']
diagram = TTKPersistenceDiagram(Input=reader)
diagram.ScalarField = ['POINTS', 'Scalars_']
keep_pairs = Threshold(Input=diagram)
keep_pairs.Scalars = ['CELLS', 'Persistence']
keep_pairs.LowerThreshold = 0.05
keep_pairs.UpperThreshold = 1.0e99
simplified = TTKTopologicalSimplification(Domain=reader, Constraints=keep_pairs)
simplified.ScalarField = ['POINTS', 'Scalars_']

# Compute the piecewise-linear Morse-Smale complex of the simplified field.
msc = TTKMorseSmaleComplex(Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.CriticalPoints = True
msc.Ascending1Separatrices = True
msc.Descending1Separatrices = True
msc.MorseSmaleComplexSegmentation = True
segmentation = OutputPort(msc, 3)

# CellDimension on critical output is 0 / 1 / 2 for min / saddle / max in 2-D.
critical = OutputPort(msc, 0)
def select_dimension(value):
    f = Threshold(Input=critical)
    f.Scalars = ['POINTS', 'CellDimension']
    f.LowerThreshold = value
    f.UpperThreshold = value
    return f
mins, saddles, maxes = map(select_dimension, (0, 1, 2))

def spheres(source):
    glyph = Glyph(Input=source, GlyphType='Sphere')
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = 'All Points'
    glyph.GlyphType.Radius = 2.0
    glyph.GlyphType.ThetaResolution = 18
    glyph.GlyphType.PhiResolution = 18
    return glyph
min_glyphs, saddle_glyphs, max_glyphs = map(spheres, (mins, saddles, maxes))
# Lift the markers infinitesimally above the 2-D field to avoid depth fighting.
def lift(source):
    f = Transform(Input=source)
    f.Transform.Translate = [0.0, 0.0, 2.1]
    return f
min_glyphs, saddle_glyphs, max_glyphs = map(lift, (min_glyphs, saddle_glyphs, max_glyphs))

view = CreateView('RenderView')
view.ViewSize = [1500, 1000]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0

seg_display = Show(segmentation, view)
ColorBy(seg_display, ('POINTS', 'MorseSmaleManifold'))
seg_display.Representation = 'Surface'
seg_display.Interpolation = 'Flat'
seg_display.Opacity = 0.92
seg_display.SetScalarBarVisibility(view, False)
lut = GetColorTransferFunction('MorseSmaleManifold')
lut.InterpretValuesAsCategories = 0
lut.ApplyPreset('Turbo', True)
lut.RescaleTransferFunction(0, 44)

# The 1-separatrices make the piecewise-linear segmentation boundaries explicit.
separatrices = OutputPort(msc, 1)
sep_display = Show(separatrices, view)
sep_display.DiffuseColor = [0.03, 0.03, 0.03]
sep_display.LineWidth = 1.5

for source, color in ((min_glyphs, [0.1, 0.25, 1.0]),
                      (saddle_glyphs, [1.0, 1.0, 1.0]),
                      (max_glyphs, [1.0, 0.05, 0.05])):
    d = Show(source, view)
    d.DiffuseColor = color
    d.AmbientColor = color
    d.ColorArrayName = [None, '']
    d.Specular = 0.35

view.CameraParallelProjection = 1
view.ResetCamera()
view.CameraParallelScale *= 1.08
SaveScreenshot('/workspace/results/fracture_morse_smale_persistence_0.05.png', view)
SaveState('/workspace/results/fracture_morse_smale_persistence_0.05.pvsm')
SaveData('/workspace/results/fracture_simplified_0.05.vti', proxy=simplified)
SaveData('/workspace/results/fracture_morse_smale_segmentation_0.05.vti', proxy=segmentation)
SaveData('/workspace/results/fracture_critical_points_0.05.vtp', proxy=critical)

# Produce a deterministic 2-D raster view of the segmentation as well.  This
# avoids graphics-driver interpolation artifacts for this planar image data.
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image, ImageDraw
import colorsys
segmentation.UpdatePipeline()
seg_data = msc.GetClientSideObject().GetOutputDataObject(3)
dims = seg_data.GetDimensions()
labels = vtk_to_numpy(seg_data.GetPointData().GetArray('MorseSmaleManifold'))
labels = labels.reshape((dims[1], dims[0]))
palette = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.61803398875) % 1, 0.62, 0.88)) for i in range(45)]
img = Image.new('RGB', (dims[0], dims[1]))
img.putdata([palette[int(v) % len(palette)] for v in labels.ravel()])
draw = ImageDraw.Draw(img)
critical.UpdatePipeline()
cp = msc.GetClientSideObject().GetOutputDataObject(0)
types = vtk_to_numpy(cp.GetPointData().GetArray('CellDimension'))
for i in range(cp.GetNumberOfPoints()):
    x, y, _ = cp.GetPoint(i)
    color = {0: (25, 65, 255), 1: (255, 255, 255), 2: (255, 15, 15)}[int(types[i])]
    r = 2
    draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline=(0, 0, 0))
img = img.resize((dims[0] * 6, dims[1] * 6), Image.Resampling.NEAREST)
img.save('/workspace/results/fracture_morse_smale_persistence_0.05.png')
