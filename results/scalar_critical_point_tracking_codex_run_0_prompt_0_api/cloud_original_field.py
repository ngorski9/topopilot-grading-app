from paraview.simple import *
from topopilot.nodes import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    args = parser.parse_args()

    scalar = LoadScalarField(args.i, "Scalars_")
    field = EmbedScalarField(scalar.id, colormap="cool warm")
    view = CreateView("RenderView")
    view.ViewSize = [1024, 1024]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.CameraPosition = [127.5, 127.5, 500.0]
    view.CameraFocalPoint = [127.5, 127.5, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 150.0
    view.InteractionMode = "2D"
    props = field.visualize(1, view)
    bar = GetScalarBar(props.LookupTable, view)
    bar.WindowLocation = "Any Location"
    bar.Position = [0.84, 0.14]
    bar.ScalarBarLength = 0.72
    bar.Title = "Scalars_ (original)"
    bar.ComponentTitle = ""
    props.SetScalarBarVisibility(view, True)
    SaveScreenshot(args.o, view)
