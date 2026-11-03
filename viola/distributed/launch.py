"""Single-node torchrun launcher for a small YAML configuration."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import read_config


def main():
    parser = argparse.ArgumentParser(prog="viola")
    parser.add_argument("command", choices=("launch",))
    parser.add_argument("--config", required=True)
    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    path = str(Path(args.config).resolve())
    config = read_config(path)
    env = dict(os.environ, VIOLA_CONFIG=path)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={config['num_processes']}", args.script, *args.script_args]
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
