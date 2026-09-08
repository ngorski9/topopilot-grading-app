from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path


def pipeline(load_path):
    # The unmodified field is retained for the background visualization.
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    field_view = EmbedScalarField(parent_node_id=scalar.id, colormap="cool warm")

    # Simplify before detecting maxima, then match those maxima through steps 1--3.
    simplified = PersistenceSimplification(parent_node_id=scalar.id, epsilon=0.5)
    maxima = ComputeScalarCriticalPoints(
        parent_node_id=simplified.id,
        which_points="maxima",
    )
    tracked = TrackScalarCriticalPoints(
        parent_node_id=maxima.id,
        time_steps=3,
        backend="emd",
    )
    tracked_view = EmbedTrackedScalarFieldCriticalPoints(
        parent_node_id=tracked.id,
        radius=2.0,
    )

    return [field_view, tracked_view], {
        field_view.id: ([0.84, 0.14], 0.72, "Scalars_"),
    }, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1024, 1024]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [127.5, 127.5, 0.0]
    view.CameraPosition = [127.5, 127.5, 500.0]
    view.CameraFocalPoint = [127.5, 127.5, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 150.0
    view.InteractionMode = "2D"
    return view


def validate_arguments(args):
    if not os.path.isdir(args.i):
        raise ValueError("Input must be the cloud time-series directory")
    if args.t not in (1, 2, 3):
        raise ValueError("This visualization is intentionally limited to time steps 1, 2, and 3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-t", type=int, required=True)
    parser.add_argument("-q", default=".")
    args = parser.parse_args()
    validate_arguments(args)

    config.set_random_seed(0)
    config.change_session_folder(args.q)
    Path(args.q).mkdir(exist_ok=True)
    Path(f"{args.q}/cache").mkdir(exist_ok=True)
    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)

    render_view = create_view()
    render_view.Background = [1.0, 1.0, 1.0]
    render_view.UseColorPaletteForBackground = 0
    for node in nodes_to_visualize:
        props = node.visualize(args.t, render_view)
        if node.id in color_bar_info:
            position, length, title = color_bar_info[node.id]
            bar = GetScalarBar(props.LookupTable, render_view)
            bar.WindowLocation = "Any Location"
            bar.Position = position
            bar.ScalarBarLength = length
            bar.Title = title
            bar.ComponentTitle = ""
            props.SetScalarBarVisibility(render_view, True)
    SaveScreenshot(args.o, render_view)
