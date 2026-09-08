from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

DATA = "/workspace/QMCPACK.vti"
OUT_PNG = "/workspace/QMCPACK_persistence_0.04.png"
OUT_STATE = "/workspace/QMCPACK_persistence_0.04.pvsm"

# Input and persistence simplification (absolute threshold, data range is [0, 1]).
volume = XMLImageDataReader(registrationName="QMCPACK", FileName=[DATA])
volume.PointArrayStatus = ["Scalars_"]
volume.UpdatePipeline()
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName="Persistence simplification (0.04)", Input=volume)
simplified.InputArray = ["POINTS", "Scalars_"]
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1
simplified.UpdatePipeline()

# Quantiles split the scalar-value samples into five equal-population regions.
data = servermanager.Fetch(simplified)
values = vtk_to_numpy(data.GetPointData().GetArray("Scalars_"))
levels = [float(np.quantile(values, q)) for q in (0.2, 0.4, 0.6, 0.8)]

contours = Contour(registrationName="Four equal-volume isocontours", Input=simplified)
contours.ContourBy = ["POINTS", "Scalars_"]
contours.Isosurfaces = levels

# PL critical points are computed from the persistence-simplified scalar field.
critical = TTKScalarFieldCriticalPoints(
    registrationName="PL critical points (simplified)", Input=simplified)
critical.ScalarField = ["POINTS", "Scalars_"]
critical.InputOffsetField = ["POINTS", "Scalars__Order"]
critical.ForceInputOffsetField = 1
critical.UpdatePipeline()

# A sphere glyph makes every point visible over the isocontours.
points = Glyph(registrationName="Critical-point spheres", Input=critical,
               GlyphType="Sphere")
points.GlyphMode = "All Points"
points.ScaleArray = ["POINTS", "No scale array"]
points.ScaleFactor = 1.35
points.GlyphType.ThetaResolution = 18
points.GlyphType.PhiResolution = 18

view = CreateView("RenderView")
view.ViewSize = [1600, 1100]
view.Background = [0.055, 0.065, 0.085]
view.BackgroundColorMode = "Gradient"
view.Background2 = [0.15, 0.18, 0.23]
view.OrientationAxesVisibility = 1

contour_display = Show(contours, view)
contour_display.Representation = "Surface"
contour_display.DiffuseColor = [0.75, 0.84, 0.92]
contour_display.Opacity = 0.27
contour_display.LineWidth = 2.5

point_display = Show(points, view)
ColorBy(point_display, ("POINTS", "CriticalType"))
lut = GetColorTransferFunction("CriticalType")
lut.InterpretValuesAsCategories = 1
lut.Annotations = ["0", "Minima", "1", "1-saddles", "2", "2-saddles", "3", "Maxima", "-1", "Multi-saddles"]
# TTK critical type: 0=min, 1=1-saddle, 2=2-saddle, 3=max; multi-saddles are white.
lut.IndexedColors = [0.10, 0.35, 1.00,  1.00, 1.00, 1.00,  1.00, 0.48, 0.05,  0.92, 0.05, 0.05,  1.00, 1.00, 1.00]
point_display.LookupTable = lut
point_display.SetScalarBarVisibility(view, True)
bar = GetScalarBar(lut, view)
bar.Title = "PL critical points\n(persistence ≥ 0.04)"
bar.ComponentTitle = ""
bar.DrawAnnotations = 1
bar.Position = [0.78, 0.10]
bar.ScalarBarLength = 0.35

# Title and isovalue annotation.
title = Text(registrationName="Title")
title.Text = "QMCPACK — persistence simplification: 0.04"
title_display = Show(title, view)
title_display.WindowLocation = "Upper Left Corner"
title_display.Color = [1, 1, 1]
title_display.FontSize = 18
iso_label = Text(registrationName="Isocontour levels")
iso_label.Text = "4 quantile isocontours (20%, 40%, 60%, 80%)\n" + ", ".join("%.4f" % x for x in levels)
iso_display = Show(iso_label, view)
iso_display.WindowLocation = "Lower Left Corner"
iso_display.Color = [0.86, 0.91, 0.96]
iso_display.FontSize = 13

view.CameraPosition = [173, -202, 169]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [-0.28, 0.36, 0.89]
view.CameraParallelScale = 105
view.CameraParallelProjection = 1
Render(view)
SaveScreenshot(OUT_PNG, view, ImageResolution=[1600, 1100])
SaveState(OUT_STATE)
print("isocontour levels:", ", ".join("%.8f" % x for x in levels))
print("wrote", OUT_PNG)
print("wrote", OUT_STATE)
