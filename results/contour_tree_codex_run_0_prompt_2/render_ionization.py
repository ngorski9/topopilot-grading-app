from paraview.simple import *
import glob, re

def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', p)]

files = sorted(glob.glob('/workspace/Ionization/*.vti'), key=natural_key)

# Time-dependent raw scalar field.
field = XMLImageDataReader(FileName=files)
field.PointArrayStatus = ['Scalars_']
field.UpdatePipeline()

# Persistence simplification: persistence diagram supplies the constraints and
# the Topological Simplification threshold is 0.1.
diagram = TTKPersistenceDiagram(Input=field)
diagram.ScalarField = ['POINTS', 'Scalars_']
# Keep only persistence pairs at or above 0.1; the retained endpoints are the
# simplification constraints.
persistence_threshold = Threshold(Input=diagram)
persistence_threshold.Scalars = ['CELLS', 'Persistence']
persistence_threshold.LowerThreshold = 0.1
persistence_threshold.UpperThreshold = 1.0e30
simplified = TTKTopologicalSimplification(Domain=field, Constraints=persistence_threshold)
simplified.ScalarField = ['POINTS', 'Scalars_']
simplified.Threshold = 0.1

# Piecewise-linear contour tree of the simplified scalar field.
tree = TTKContourTree(Input=simplified)
tree.ScalarField = ['POINTS', 'Scalars_']
# Sample each arc at its piecewise-linear vertices.
tree.ArcSampling = 1

view = CreateView('RenderView')
view.ViewSize = [1400, 1000]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0

# Raw field, shown as a semi-transparent viridis-colored volume.
field_display = Show(field, view)
field_display.Representation = 'Volume'
ColorBy(field_display, ('POINTS', 'Scalars_'))
field_lut = GetColorTransferFunction('Scalars_')
field_lut.ApplyPreset('Viridis (matplotlib)', True)
field_lut.RescaleTransferFunction(0.0, 37.44280364288978)
field_display.LookupTable = field_lut
field_display.Opacity = 0.08
field_display.ScalarOpacityUnitDistance = 1.0
field_display.SetScalarBarVisibility(view, True)
field_bar = GetScalarBar(field_lut, view)
field_bar.Title = 'Raw Scalars_'
field_bar.ComponentTitle = ''
field_bar.LabelColor = [0.9, 0.9, 0.9]
field_bar.TitleColor = [0.9, 0.9, 0.9]

# Contour-tree arcs: output port 1, rendered as radius-1 tubes.
arcs = OutputPort(tree, 1)
tubes = Tube(Input=arcs)
tubes.Radius = 1.0
tubes.NumberofSides = 10
arc_display = Show(tubes, view)
arc_display.DiffuseColor = [0.92, 0.92, 0.92]
arc_display.AmbientColor = [0.92, 0.92, 0.92]
arc_display.Ambient = 0.5

# Contour-tree critical points: output port 0.  TTK uses 0=min, 1=1-saddle,
# 2=2-saddle, 3=max; each category gets radius-2 sphere glyphs.
nodes = OutputPort(tree, 0)
critical = [
    ('minima', 0, [0.10, 0.35, 1.00]),
    ('one_saddles', 1, [1.00, 1.00, 1.00]),
    ('two_saddles', 2, [1.00, 0.50, 0.00]),
    ('maxima', 3, [1.00, 0.05, 0.05]),
]
for name, value, color in critical:
    selected = Threshold(Input=nodes)
    selected.Scalars = ['POINTS', 'CriticalType']
    selected.LowerThreshold = value
    selected.UpperThreshold = value
    sphere = Sphere(Radius=2.0, ThetaResolution=16, PhiResolution=12)
    glyph = Glyph(Input=selected, GlyphType=sphere)
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphTransform = 'Transform2'
    display = Show(glyph, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.3

tree.UpdatePipeline()
Render(view)
view.ResetCamera()
view.CameraPosition = [322.0, -430.0, 315.0]
view.CameraFocalPoint = [120.0, 120.0, 80.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelProjection = 0
Render(view)

SaveScreenshot('/workspace/ionization_contour_tree.png', view, ImageResolution=[1400, 1000])
SaveState('/workspace/ionization_contour_tree.pvsm')
