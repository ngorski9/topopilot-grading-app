"""ParaView/TTK visualization for the time-varying Ionization data set.

Run with: pvpython visualize_ionization.py
It writes Ionization/Ionization.pvd, ionization_contour_tree.pvsm and two PNGs.
"""
from pathlib import Path
from paraview.simple import *

base = Path('/workspace/Ionization')
files = sorted(base.glob('Ionization*.vti'), key=lambda p: int(p.stem.replace('Ionization', '')))
pvd = base / 'Ionization.pvd'
pvd.write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
               '  <Collection>\n' + ''.join(
                   f'    <DataSet timestep="{i}" group="" part="0" file="{f.name}"/>\n'
                   for i, f in enumerate(files)) + '  </Collection>\n</VTKFile>\n')

data = PVDReader(FileName=str(pvd))

# Topological persistence simplification (absolute persistence = 0.1), followed
# by a contour tree of the resulting piecewise-linear scalar field.
simplified = TTKTopologicalSimplificationByPersistence(Input=data)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.1
simplified.ThresholdIsAbsolute = 1
simplified.PairType = 0

tree = TTKContourTree(Input=simplified)
tree.ScalarField = ['POINTS', 'Scalars_']
tree.ArcSampling = 0

# Two linked views: original field (a central slice) and the simplified tree.
layout = CreateLayout('Ionization layout')
field_view = CreateView('RenderView')
tree_view = CreateView('RenderView')
AssignViewToLayout(view=field_view, layout=layout, hint=0)
AssignViewToLayout(view=tree_view, layout=layout, hint=2)
for view in (field_view, tree_view):
    view.ViewSize = [900, 720]
    view.Background = [0.08, 0.08, 0.10]

field_slice = Slice(Input=data)
field_slice.SliceType = 'Plane'
field_slice.SliceType.Origin = [149.5, 61.5, 61.5]
field_slice.SliceType.Normal = [0.0, 0.0, 1.0]
field_display = Show(field_slice, field_view)
ColorBy(field_display, ('POINTS', 'Scalars_'))
field_lut = GetColorTransferFunction('Scalars_')
field_lut.ApplyPreset('Viridis (matplotlib)', True)
field_display.RescaleTransferFunctionToDataRange(True, False)
field_display.SetScalarBarVisibility(field_view, True)
field_display.Representation = 'Surface'

# The tree's arc output is rendered as cylindrical edges of radius 1.
arc_lines = ExtractSurface(Input=OutputPort(tree, 1))
arcs = Tube(Input=arc_lines)
arcs.Radius = 1.0
arcs.NumberofSides = 12
arc_display = Show(arcs, tree_view)
arc_display.DiffuseColor = [0.72, 0.72, 0.72]

# CriticalType values emitted by TTK: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
def critical_points(value, color, name):
    selected = Threshold(Input=OutputPort(tree, 0))
    selected.Scalars = ['POINTS', 'CriticalType']
    selected.LowerThreshold = value
    selected.UpperThreshold = value
    selected.ThresholdMethod = 'Between'
    glyph = Glyph(Input=selected, GlyphType='Sphere')
    glyph.GlyphType.Radius = 2.0
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = 'All Points'
    display = Show(glyph, tree_view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.25
    display.Specular = 0.15
    RenameSource(name, glyph)
    return glyph

critical_points(3, [1.0, 0.0, 0.0], 'Maxima (red, radius 2)')
critical_points(2, [1.0, 0.45, 0.0], '2-saddles (orange, radius 2)')
critical_points(1, [1.0, 1.0, 1.0], '1-saddles (white, radius 2)')
critical_points(0, [0.10, 0.35, 1.0], 'Minima (blue, radius 2)')

field_view.CameraPosition = [150, -420, 300]
field_view.CameraFocalPoint = [150, 62, 62]
field_view.CameraViewUp = [0, 0, 1]
tree_view.CameraPosition = [430, -430, 330]
tree_view.CameraFocalPoint = [150, 62, 62]
tree_view.CameraViewUp = [0, 0, 1]

# First time step is deliberately used as the representative frame while the
# PVD reader retains all 21 time steps for animation in ParaView.
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.AnimationTime = 0
RenderAllViews()
SaveScreenshot('/workspace/ionization_original_and_tree.png', layout, SaveAllViews=1, ImageResolution=[1800, 720])
SaveState('/workspace/ionization_contour_tree.pvsm')

# Tree-only image makes the requested simplified contour tree easy to inspect.
Hide(field_slice, field_view)
Render(tree_view)
SaveScreenshot('/workspace/ionization_simplified_contour_tree.png', tree_view, ImageResolution=[1000, 800])
print(f'Loaded {len(files)} time steps; wrote {pvd}')
