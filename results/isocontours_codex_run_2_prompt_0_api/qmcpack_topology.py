from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    simplified = PersistenceSimplification(
        parent_node_id=scalar.id,
        epsilon=0.04,
    )

    # Four levels split the scalar-value range into five equal intervals.
    contours = ComputeIsocontours(
        parent_node_id=simplified.id,
        num_contours=4,
    )
    critical_points = ComputeScalarCriticalPoints(
        parent_node_id=simplified.id,
        which_points="all critical points (mins+maxes+saddles)",
    )

    field_view = EmbedScalarField(
        parent_node_id=simplified.id,
        colormap="cool warm",
    )
    contour_view = EmbedIsocontours(
        parent_node_id=contours.id,
        colormap="black body",
        tube_radius=1.0,
    )
    critical_point_view = EmbedScalarCriticalPoints(
        parent_node_id=critical_points.id,
        radius=2.5,
        # minima, 1-saddles, 2-saddles, maxima, degenerate points
        color_scheme=["#0000ff", "#ffffff", "#ffa500", "#ff0000", "#000000"],
    )

    return [field_view, contour_view, critical_point_view], {
        field_view.id: ([0.84, 0.14], 0.68, "Scalars_ (simplified)"),
    }, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 960]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [34.0, 34.0, 57.0]
    view.CameraPosition = [185.0, 175.0, 205.0]
    view.CameraFocalPoint = [34.0, 34.0, 57.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 125.0
    view.InteractionMode = "3D"
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
    if args.t > time_steps or (args.o == "" and not args.x) or (args.o and args.x):
        print("error: invalid arguments")
        exit(1)
    if args.t == -1 and time_steps == 1:
        args.t = 1
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-x", action=argparse.BooleanOptionalAction)
    parser.add_argument("-o", default="")
    parser.add_argument("-t", type=int, default=-1)
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
        props = node.visualize(1, render_view)
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
