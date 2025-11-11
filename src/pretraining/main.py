import argparse
import logging
import os
import torch
import torch.distributed as dist
from src.pretraining.vita_trainer import vita_training_loop
from src.utils.utils import setup_distributed, cleanup_distributed, setup_logging
from src.utils.utils import parse_args

parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume-from-checkpoint",
    help="path to resume from checkpoint",
    default=None,
    type=str,
)
parser.add_argument(
    "--pretrained-model-path",
    help="path to pretrained model to load before training",
    default=None,
    type=str,
)
parser.add_argument("--batch-size", help="batch size", default=256, type=int)
parser.add_argument(
    "--n-masked-features",
    help="number of masked features, the rest of the features are input features",
    default=10,
    type=int,
)
parser.add_argument(
    "--n-epochs", help="number of training epochs", default=100, type=int
)
parser.add_argument(
    "--init-lr", help="initial learning rate", default=0.0005, type=float
)
parser.add_argument(
    "--n-warmup-epochs", help="number of warm-up epochs", default=10, type=float
)
parser.add_argument(
    "--decay-factor",
    help="exponential learning rate decay factor after warmup",
    default=0.99,
    type=float,
)
parser.add_argument(
    "--model-size",
    help="model size mini (60k), small (2M), medium (8M), and large (56M)",
    default="small",
    type=str,
)
parser.add_argument(
    "--beta",
    help="parameter for sinusoidal prior loss weighting",
    default=0.5,
    type=float,
)


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Setup logging
    setup_logging(rank)

    try:
        args_dict = parse_args(parser)

        # Add distributed training info to args
        args_dict["rank"] = rank
        args_dict["world_size"] = world_size
        args_dict["local_rank"] = local_rank
        vita_training_loop(args_dict)
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()
