"""
Main script for Self-Supervised Learning (SSL) pre-training.
"""

__author__ = "Houtan Ghaffari"
__email__ = "houtan.ghaffari@gmail.com"
__version__ = "1.0.0"

from yaml_config import get_ssl_config
from engine.utils import seed_everything
from engine.ssl_trainer import run_ssl_experiment
from models.vit import Attention


def main() -> None:
    seed_everything()
    args = get_ssl_config()
    Attention.fused = not args.explicit_attn
    run_ssl_experiment(args)


if __name__ == '__main__':
    main()

