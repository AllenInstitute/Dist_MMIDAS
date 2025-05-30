#!/usr/bin/env python3

from argparse import ArgumentParser, RawDescriptionHelpFormatter

LOGO = [
    "   _____         _ __       __  _    _____    ______",
    "  / ___/      __(_) /______/ /_| |  / /   |  / ____/",
    "  \\__ \\ | /| / / / __/ ___/ __ \\ | / / /| | / __/",
    " ___/ / |/ |/ / / /_/ /__/ / / / |/ / ___ |/ /___",
    "/____/|__/|__/_/\\__/\\___/_/ /_/|___/_/  |_/_____/"
]

def main():
    parser = ArgumentParser(prog="switch",
                            description="Scale up VAE training with efficient sparsity.",
                            epilog="\n".join(LOGO),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("config",
                        help="Name of the config file (.py) to launch an " +
                             "experiment. Expected to be found in the " +
                             "config/ directory.",
                        type=str)
    args = parser.parse_args()

    print("hello switch")

if __name__ == "__main__":
    main()
