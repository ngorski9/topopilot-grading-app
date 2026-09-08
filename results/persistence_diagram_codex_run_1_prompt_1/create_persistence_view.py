from paraview.simple import *

data_path = "/workspace/QMCPACK.vti"

reader = XMLImageDataReader(registrationName="QMCPACK scalar field", FileName=[data_path])
reader.PointArrayStatus = ["Scalars_"]

simplified = TTKTopologicalSimplificationByPersistence(
    registrationName="Persistence simplification (0.04)", Input=reader)
simplified.InputArray = ["POINTS", "Scalars_"]
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1

diagram = TTKPersistenceDiagram(
    registrationName="Piecewise linear persistence diagram (sublevel)", Input=simplified)
diagram.ScalarField = ["POINTS", "Scalars_"]
# TTK's standard persistence-diagram filter computes the piecewise-linear,
# sublevel-set diagram for a scalar field.

order0 = Threshold(registrationName="Persistence pairs — order 0", Input=diagram)
order0.Scalars = ["CELLS", "PairType"]
order0.LowerThreshold = 0.0
order0.UpperThreshold = 0.0

order2 = Threshold(registrationName="Persistence pairs — order 2", Input=diagram)
order2.Scalars = ["CELLS", "PairType"]
order2.LowerThreshold = 2.0
order2.UpperThreshold = 2.0

view = CreateView("RenderView")
view.ViewSize = [1200, 800]
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0

display0 = Show(order0, view)
display0.Representation = "Surface"
display0.DiffuseColor = [0.12, 0.35, 0.85]
display0.LineWidth = 3.0
display0.PointSize = 8.0

display2 = Show(order2, view)
display2.Representation = "Surface"
display2.DiffuseColor = [0.9, 0.2, 0.15]
display2.LineWidth = 3.0
display2.PointSize = 8.0

annotation = Text(registrationName="Diagram description")
annotation.Text = "Piecewise-linear persistence diagram | sublevel-set filtration | simplified at 0.04 | blue: order 0, red: order 2"
annotation_display = Show(annotation, view)
annotation_display.WindowLocation = "Upper Center"
annotation_display.FontSize = 16
annotation_display.Color = [0.0, 0.0, 0.0]

SetActiveView(view)
Render()
view.ResetCamera()
Render()
SaveScreenshot("/workspace/QMCPACK_persistence_diagram.png", view)
SaveState("/workspace/QMCPACK_persistence_diagram.pvsm")
