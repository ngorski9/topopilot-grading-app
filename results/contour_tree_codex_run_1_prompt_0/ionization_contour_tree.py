from paraview.simple import *
from paraview import servermanager
import glob

files = sorted(glob.glob('/workspace/Ionization/Ionization*.vti'),
               key=lambda p: int(p.split('Ionization')[-1].split('.vti')[0]))

# Load the supplied VTI files as a time-varying series; the saved image shows
# its first time step, and the state file retains all 21 time steps.
data = XMLImageDataReader(FileName=files)
data.PointArrayStatus = ['Scalars_']

# Persistence simplification, retaining features above 0.1 of scalar range.
simplified = TTKTopologicalSimplificationByPersistence(Input=data)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PairType = 'Extremum-Saddle'
simplified.PersistenceThreshold = 0.1
simplified.ThresholdIsAbsolute = 0

tree = TTKContourTree(Input=simplified)
tree.ScalarField = ['POINTS', 'Scalars_']

# TTK ports: 0 = nodes, 1 = arcs, 2 = segmentation.
nodes = OutputPort(tree, 0)
arcs = OutputPort(tree, 1)

# Edge radius 1.
edge_surface = ExtractSurface(Input=arcs)
edge_tubes = Tube(Input=edge_surface)
edge_tubes.Radius = 1.0
edge_tubes.NumberofSides = 12

# Separate critical-point classes (TTK: min=0, 1-saddle=1, 2-saddle=2, max=3).
def critical_points(value):
    lo = Threshold(Input=nodes)
    lo.Scalars = ['POINTS', 'CriticalType']
    lo.LowerThreshold = value
    lo.UpperThreshold = value
    lo.ThresholdMethod = 'Between'
    glyph = Glyph(Input=lo, GlyphType='Sphere')
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.GlyphType.Radius = 2.0
    glyph.GlyphType.ThetaResolution = 18
    glyph.GlyphType.PhiResolution = 18
    return glyph

minima = critical_points(0)
saddle1 = critical_points(1)
saddle2 = critical_points(2)
maxima = critical_points(3)

# Left: original scalar field.  Right: simplified piecewise-linear contour tree.
slice1 = Slice(Input=data)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = [149.5, 61.5, 61.5]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

v1 = CreateRenderView()
v1.ViewSize = [900, 760]
v1.Background = [0.08, 0.08, 0.10]
v1.UseColorPaletteForBackground = 0
v1.CameraPosition = [149.5, 61.5, 600]
v1.CameraFocalPoint = [149.5, 61.5, 61.5]
v1.CameraParallelProjection = 1

v2 = CreateRenderView()
v2.ViewSize = [900, 760]
v2.Background = [0.08, 0.08, 0.10]
v2.UseColorPaletteForBackground = 0
v2.CameraParallelProjection = 1

field = Show(slice1, v1)
ColorBy(field, ('POINTS', 'Scalars_'))
lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Viridis (matplotlib)', True)
field.RescaleTransferFunctionToDataRange(True, False)
bar = GetScalarBar(lut, v1)
bar.Title = 'Ionization scalar'
bar.ComponentTitle = ''
bar.Visibility = 1

e = Show(edge_tubes, v2)
e.DiffuseColor = [0.72, 0.72, 0.72]

def show_points(source, color):
    rep = Show(source, v2)
    rep.DiffuseColor = color
    rep.AmbientColor = color
    rep.Ambient = 0.25
    return rep

show_points(maxima, [1.0, 0.0, 0.0])
show_points(saddle2, [1.0, 0.5, 0.0])
show_points(saddle1, [1.0, 1.0, 1.0])
show_points(minima, [0.05, 0.25, 1.0])

v1.ResetCamera()
v2.ResetCamera()
v2.CameraPosition = [150, -430, 90]
v2.CameraFocalPoint = [150, 60, 16]
v2.CameraViewUp = [0, 0, 1]

layout_views = CreateLayout('Ionization contour tree')
layout_views.AssignView(0, v1)
layout_views.SplitHorizontal(0, 0.5)
layout_views.AssignView(2, v2)
SaveScreenshot('/workspace/ionization_contour_tree.png', layout_views, ImageResolution=[1800,760])
SaveState('/workspace/ionization_contour_tree.pvsm')

# Persist useful geometry for non-interactive inspection.
SaveData('/workspace/simplified_contour_tree_edges.vtp', proxy=edge_tubes)
SaveData('/workspace/simplified_contour_tree_nodes.vtu', proxy=nodes)
print('Saved screenshot, ParaView state, and simplified contour-tree geometry.')
