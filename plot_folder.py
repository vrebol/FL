import argparse
import utils.plots as plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FL Training')
    parser.add_argument("--session_tag", type=str, default="folder_plots", help="Sets name of subfolder for experiments")

    args = parser.parse_args()

    path_prefix = "runs/" + args.session_tag + "/"
   
    plots.plot_config_folder(path_prefix)

