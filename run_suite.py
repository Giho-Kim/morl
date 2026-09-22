"""Sequential launcher: each environment x method x seed is a separate process."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=["fruit_tree", "minecart", "mo_hopper", "mo_ant"],
                        choices=["fruit_tree", "minecart", "mo_hopper", "mo_ant"])
    parser.add_argument("--methods", nargs="+", default=["uniform", "d"],
                        choices=["uniform", "d", "a", "e", "td"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="runs")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    smoke = (["--steps", "160", "--learning-starts", "16", "--batch-size", "16",
              "--hidden", "32", "--eval-every", "160", "--eval-tasks", "3",
              "--eval-episodes", "1", "--horizon", "12", "--replay-size", "512",
              "--start-samples", "2"] if args.smoke else [])
    for env in args.envs:
        for method in args.methods:
            for seed in args.seeds:
                cmd = [sys.executable, str(Path(__file__).with_name("train.py")),
                       "--env", env, "--method", method, "--seed", str(seed),
                       "--output", args.output, *smoke, *extra]
                print(" ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
