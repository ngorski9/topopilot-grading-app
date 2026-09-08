from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    # Render the original field; simplify only the branch used for topological analysis.
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    scalar_view = EmbedScalarField(parent_node_id=scalar.id, colormap="cool warm")

    simplified = PersistenceSimplification(parent_node_id=scalar.id, epsilon=0.5)
    maxima = ComputeScalarCriticalPoints(parent_node_id=simplified.id, which_points="maxima")
    tracked_maxima = TrackScalarCriticalPoints(
        parent_node_id=maxima.id, time_steps=3, backend="emd"
    )
    tracked_maxima_view = EmbedTrackedScalarFieldCriticalPoints(
        parent_node_id=tracked_maxima.id, radius=2.0
    )

    return [scalar_view, tracked_maxima_view], {
        scalar_view.id: ([0.84, 0.15], 0.70, "Scalars_"),
    }, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 1024]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [127.5, 127.5, 0.0]
    view.CameraPosition = [127.5, 127.5, 500.0]
    view.CameraFocalPoint = [127.5, 127.5, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 145.0
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
                print(f"missing file {dirname}/{basename}/{basename}{i}.vti")
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
    if args.t == -1:
        raise ValueError("This pipeline tracks exactly 3 time steps; select -t 1, 2, or 3.")
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
