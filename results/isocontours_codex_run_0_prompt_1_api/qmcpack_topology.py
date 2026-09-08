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
    simplified = PersistenceSimplification(parent_node_id=scalar.id, epsilon=0.04)

    field_view = EmbedScalarField(parent_node_id=simplified.id, colormap="viridis")
    contours = ComputeIsocontours(parent_node_id=simplified.id, num_contours=4)
    contour_view = EmbedIsocontours(
        parent_node_id=contours.id,
        colormap="black body",
        tube_radius=1.0,
    )
    critical_points = ComputeScalarCriticalPoints(
        parent_node_id=simplified.id,
        which_points="all critical points (mins+maxes+saddles)",
    )
    critical_point_view = EmbedScalarCriticalPoints(
        parent_node_id=critical_points.id,
        radius=2.5,
        # minima, index-1 saddles, index-2 saddles, maxima, degenerate
        color_scheme=["#0000ff", "#ffffff", "#ff8000", "#ff0000", "#000000"],
    )

    return [field_view, contour_view, critical_point_view], {
        field_view.id: ([0.84, 0.15], 0.65, "Simplified Scalars_"),
    }, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 900]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [34.0, 34.0, 57.0]
    view.CameraPosition = [185.0, -205.0, 175.0]
    view.CameraFocalPoint = [34.0, 34.0, 57.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 165.0
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
    if args.t > time_steps:
        print("error: time step is too high")
        exit(1)
    if args.x and args.t == -1 and time_steps > 1:
        print("error: cannot run interactive mode on multiple time steps")
        exit(1)
    if args.o != "" and args.x:
        print("error: you cannot specify an output file and -interactive")
        exit(1)
    if args.o == "" and not args.x:
        print("error: you must either specify an output file or run in interactive mode.")
        exit(1)
    if args.t == -1 and time_steps > 1 and args.o[-4:] != ".mp4":
        print("error: when outputting to video format your output filename must end in .mp4")
        exit(1)
    if args.t != -1 and not args.x and args.o[-4:] != ".png":
        print("error: when outputting to image format your output filename must end in .png")
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
    time_steps = validate_arguments(args)
    config.set_random_seed(args.seed)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)
    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)
    renderView = create_view()
    renderView.Background = [1.0, 1.0, 1.0]
    renderView.UseColorPaletteForBackground = 0
    for n in nodes_to_visualize:
        props = n.visualize(args.t, renderView)
        if n.id in color_bar_info:
            position, length, title = color_bar_info[n.id]
            bar = GetScalarBar(props.LookupTable, renderView)
            bar.WindowLocation = "Any Location"
            bar.Position = position
            bar.ScalarBarLength = length
            bar.Title = title
            bar.ComponentTitle = ""
            props.SetScalarBarVisibility(renderView, True)
    if args.o:
        SaveScreenshot(args.o, renderView)
    else:
        Interact()
