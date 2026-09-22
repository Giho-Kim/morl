"""Run a saved policy on fixed held-out tasks without fine-tuning."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from core import SFSAC
from train import Config, evaluate, evaluation_tasks, initial_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("heldout_eval.npz"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    cfg = Config(**saved["config"])
    cfg.device, cfg.eval_tasks, cfg.eval_episodes = args.device, args.tasks, args.episodes
    cfg.eval_seed = args.seed
    cfg.resolve()
    torch.set_num_threads(cfg.threads)
    if saved.get("algorithm") not in ("conditional_sf_sac_v1",
                                      "conditional_sf_sac_v2_auto_temperature"):
        raise ValueError("This evaluator requires a new conditional SF-SAC checkpoint")
    temperature = float(saved.get("temperature", cfg.temperature))
    net = SFSAC(saved["obs_dim"], saved["actions"], saved["dim"], cfg.hidden,
                saved["discrete"], saved["low"], saved["high"], temperature).to(args.device)
    net.load_state_dict(saved["model"])
    net.eval()
    tasks = evaluation_tasks(cfg, saved["dim"])
    metrics, arrays = evaluate(net, cfg, tasks, initial_bank(cfg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metrics_json=json.dumps(metrics))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
