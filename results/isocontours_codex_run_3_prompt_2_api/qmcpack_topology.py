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

    contours = ComputeIsocontours(parent_node_id=simplified.id, num_contours=4)
    contour_view = EmbedIsocontours(
        parent_node_id=contours.id,
        colormap="viridis",
    )

    critical_points = ComputeScalarCriticalPoints(
        parent_node_id=simplified.id,
        which_points="all critical points (mins+maxes+saddles)",
    )
    critical_point_view = EmbedScalarCriticalPoints(
        parent_node_id=critical_points.id,
        radius=2.0,
        # minima, 1-saddles, 2-saddles, maxima, degenerate
        color_scheme=["#0000ff", "#ffffff", "#ffa500", "#ff0000", "#000000"],
    )

    return [contour_view, critical_point_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1600, 1000]
    view.OrientationAxesVisibility = 1
    view.CameraPosition = [175.0, -205.0, 180.0]
    view.CameraFocalPoint = [34.0, 34.0, 57.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = False
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
                print(f"error: missing file {dirname}/{basename}/{basename}{i}.vti")
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
    render_view.Background = [0.08, 0.08, 0.10]
    render_view.UseColorPaletteForBackground = 0
    for index, node in enumerate(nodes_to_visualize):
        properties = node.visualize(args.t, render_view)
        # Keep the four nested contour surfaces readable without obscuring the
        # complete critical-point set inside the volume.
        if index == 0:
            properties.Opacity = 0.38
    if args.o:
        SaveScreenshot(args.o, render_view)
    else:
        Interact()
