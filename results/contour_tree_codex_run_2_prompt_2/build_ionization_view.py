from paraview.simple import *
import glob
import os
import re

work = "/workspace"
files = sorted(glob.glob(os.path.join(work, "Ionization", "Ionization*.vti")),
               key=lambda p: int(re.search(r"(\d+)\.vti$", p).group(1)))

# 21 VTI files are opened as one time-dependent source.
raw = XMLImageDataReader(registrationName="Ionization (raw time series)", FileName=files)

# Simplify the scalar field first, then compute the contour tree of the result.
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName="Persistence simplification (0.1)", Input=raw)
simplified.InputArray = ["POINTS", "Scalars_"]
simplified.PersistenceThreshold = 0.1
simplified.ThresholdIsAbsolute = 1

tree = TTKContourTree(registrationName="Simplified piecewise-linear contour tree", Input=simplified)
tree.ScalarField = ["POINTS", "Scalars_"]
tree.ArcSampling = 0

view = CreateView("RenderView")
view.ViewSize = [1400, 900]
view.Background = [0.08, 0.08, 0.08]

# Raw scalar field, coloured with the requested viridis map.
raw_display = Show(raw, view)
ColorBy(raw_display, ("POINTS", "Scalars_"))
raw_lut = GetColorTransferFunction("Scalars_")
raw_lut.ApplyPreset("Viridis (matplotlib)", True)
raw_display.Representation = "Volume"
raw_display.Opacity = 0.10
raw_display.RescaleTransferFunctionToDataRange(True, False)
raw_display.SetScalarBarVisibility(view, True)

# TTK contour-tree output port 1 carries the arcs; tube radius is exactly 1.
arcs = OutputPort(tree, 1)
# TTK stores arcs in an unstructured grid; extract its polydata surface before tubing.
arc_polydata = ExtractSurface(registrationName="Contour tree edge polydata", Input=arcs)
tubes = Tube(registrationName="Contour tree edges (radius 1)", Input=arc_polydata)
tubes.Radius = 1.0
tubes.NumberofSides = 12
edge_display = Show(tubes, view)
edge_display.DiffuseColor = [0.95, 0.95, 0.95]
edge_display.AmbientColor = [0.95, 0.95, 0.95]
edge_display.Ambient = 1.0

# Output port 0 holds contour-tree nodes. CriticalType codes are:
# 0=minima, 1=1-saddles, 2=2-saddles, 3=maxima.
nodes = OutputPort(tree, 0)
def show_critical_points(name, critical_type, color):
    selected = Threshold(registrationName=name + " selection", Input=nodes)
    selected.Scalars = ["POINTS", "CriticalType"]
    selected.ThresholdMethod = "Between"
    selected.LowerThreshold = critical_type
    selected.UpperThreshold = critical_type
    glyph = Glyph(registrationName=name + " (radius 2)", Input=selected,
                  GlyphType="Sphere")
    glyph.GlyphType.Radius = 2.0
    glyph.ScaleArray = ["POINTS", "No scale array"]
    glyph.ScaleFactor = 1.0
    display = Show(glyph, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.4

show_critical_points("Minima", 0, [0.0, 0.25, 1.0])
show_critical_points("1-saddles", 1, [1.0, 1.0, 1.0])
show_critical_points("2-saddles", 2, [1.0, 0.5, 0.0])
show_critical_points("Maxima", 3, [1.0, 0.0, 0.0])

view.ResetCamera()
view.CameraPosition = [150, -480, 360]
view.CameraFocalPoint = [150, 62, 62]
view.CameraViewUp = [0, 0, 1]
Render(view)

SaveScreenshot(os.path.join(work, "Ionization_contour_tree.png"), view, ImageResolution=[1400, 900])
SaveState(os.path.join(work, "Ionization_contour_tree.pvsm"))
