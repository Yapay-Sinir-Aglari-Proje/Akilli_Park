"""PPO eğitimi — `train_rl.py --algo ppo` kısayolu."""

from __future__ import annotations

import argparse

from config import RL_N_ENVS, RL_TOTAL_TIMESTEPS, RANDOM_SEED
from train_rl import train_algo
from utils.seeds import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid navigasyon PPO eğitimi")
    parser.add_argument("--timesteps", type=int, default=RL_TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=max(1, RL_N_ENVS))
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)
    train_algo("ppo", args.timesteps, args.n_envs)


if __name__ == "__main__":
    main()
