from pathlib import Path
import hydra
from datetime import datetime
from omegaconf import OmegaConf
import os
import wandb
import torch
import numpy as np
import random

from common.io_utils import make_dir, save_yaml
from trainer import build_trainer


@hydra.main(version_base=None, config_path="./config", config_name="default")
def main(cfg):
    if not cfg.exp_dir:
        cfg.exp_dir = cfg.base_dir + '/' + f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" 
        
    make_dir(cfg.exp_dir)
    OmegaConf.save(config=cfg, f=Path(cfg.exp_dir) / "config.yaml")
    trainer = build_trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()