from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    # Preserve the original field for the background visualization.
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    field_view = EmbedScalarField(parent_node_id=scalar.id, colormap="cool warm")

    # Simplify before extracting and tracking piecewise-linear maxima.
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
        field_view.id: ([0.84, 0.15], 0.70, "Scalars_"),
    }, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 720]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [127.5, 127.5, 0.0]
    view.CameraPosition = [127.5, 127.5, 1000.0]
    view.CameraFocalPoint = [127.5, 127.5, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelScale = 150.0
    view.CameraParallelProjection = True
    view.InteractionMode = "2D"
    return view


def validate_arguments(args):
    if args.i[-4:] == ".vti":
        time_steps = 1
    else:
        if not os.path.isdir(args.i):
            print("error: file specified with -i must either be .vti file or directory")
            exit(1)
        time_steps = len(os.listdir(args.i))
        basename = os.path.basename(args.i)
        dirname = os.path.dirname(args.i)
        for i in range(1, time_steps + 1):
            if not os.path.isfile(f"{dirname}/{basename}/{basename}{i}.vti"):
                print("error: invalid input format for directory")
                exit(1)
    if args.t > time_steps:
        print("error: time step is too high")
        exit(1)
    if args.t == -1:
        # This pipeline tracks exactly three steps, so only render those frames.
        args.t = 1
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", default="")
    parser.add_argument("-t", type=int, default=1)
    parser.add_argument("-q", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    validate_arguments(args)
    config.set_random_seed(args.seed)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)

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
    if args.o:
        SaveScreenshot(args.o, render_view)
    else:
        Interact()
