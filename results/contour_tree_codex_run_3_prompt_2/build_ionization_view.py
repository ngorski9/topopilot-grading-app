from paraview.simple import *
import glob, os

# Keep the full directory-backed VTI series: ParaView exposes its files as timesteps.
files = sorted(glob.glob('/workspace/Ionization/Ionization*.vti'),
               key=lambda p: int(os.path.basename(p)[10:-4]))
series = XMLImageDataReader(registrationName='Ionization time series', FileName=files)
series.PointArrayStatus = ['Scalars_']

# Persistence simplification (0.1) followed by the piecewise-linear contour tree.
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.1)', Input=series)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.1
simplified.ThresholdIsAbsolute = 1
simplified.PairType = 'Extremum-Saddle'

tree = TTKContourTree(registrationName='Simplified piecewise-linear contour tree', Input=simplified)
tree.ScalarField = ['POINTS', 'Scalars_']
tree.ArcSampling = 0

# Two complementary views: scalar data at left, topology at right.
layout = CreateLayout('Ionization scalar field and contour tree')
raw_view = CreateView('RenderView')
tree_view = CreateView('RenderView')
layout.SplitHorizontal(0, 0.5)
layout.AssignView(1, raw_view)
layout.AssignView(2, tree_view)
raw_view.ViewSize = [900, 700]
tree_view.ViewSize = [900, 700]
for v in (raw_view, tree_view):
    v.Background = [0.12, 0.12, 0.14]

# Raw scalar field, shown as a central slice in Viridis for a readable field view.
raw_slice = Slice(registrationName='Raw scalar field (central slice)', Input=series)
raw_slice.SliceType = 'Plane'
raw_slice.SliceType.Origin = [149.5, 61.5, 61.5]
raw_slice.SliceType.Normal = [0, 0, 1]
raw_display = Show(raw_slice, raw_view, 'GeometryRepresentation')
ColorBy(raw_display, ('POINTS', 'Scalars_'))
raw_lut = GetColorTransferFunction('Scalars_')
raw_lut.ApplyPreset('Viridis (matplotlib)', True)
raw_display.SetScalarBarVisibility(raw_view, True)

# Contour-tree arcs: output port 1, converted to tubes of radius 1.
arcs = OutputPort(tree, 1)
# TTK emits an unstructured grid; extract its line geometry before tubing.
arc_lines = ExtractSurface(registrationName='Contour tree piecewise-linear arcs', Input=arcs)
tubes = Tube(registrationName='Contour tree edges (radius 1)', Input=arc_lines)
tubes.Radius = 1.0
tubes.NumberofSides = 12
arc_display = Show(tubes, tree_view, 'GeometryRepresentation')
arc_display.DiffuseColor = [0.70, 0.70, 0.70]

# Critical-node groups use TTK's standard CriticalType encoding:
# minimum=0, saddle1=1, saddle2=2, maximum=3.
nodes = OutputPort(tree, 0)
def critical_nodes(name, lo, hi, color):
    threshold = Threshold(registrationName=name, Input=nodes)
    threshold.Scalars = ['POINTS', 'CriticalType']
    threshold.LowerThreshold = lo
    threshold.UpperThreshold = hi
    threshold.ThresholdMethod = 'Between'
    glyph = Glyph(registrationName=name + ' (radius 2)', Input=threshold, GlyphType='Sphere')
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 2.0
    glyph.GlyphType.Radius = 1.0
    display = Show(glyph, tree_view, 'GeometryRepresentation')
    display.DiffuseColor = color
    return glyph

critical_nodes('Minima', 0, 0, [0.0, 0.25, 1.0])
critical_nodes('1-saddles', 1, 1, [1.0, 1.0, 1.0])
critical_nodes('2-saddles', 2, 2, [1.0, 0.50, 0.0])
critical_nodes('Maxima', 3, 3, [1.0, 0.0, 0.0])

raw_view.ResetCamera()
tree_view.ResetCamera()
tree_view.CameraPosition = [149.5, -400, 260]
tree_view.CameraFocalPoint = [149.5, 61.5, 61.5]
tree_view.CameraViewUp = [0, 0, 1]

SaveState('/workspace/Ionization_contour_tree.pvsm')
RenderAllViews()
SaveScreenshot('/workspace/Ionization_contour_tree.png', layout, SaveAllViews=1, ImageResolution=[1800, 700])
