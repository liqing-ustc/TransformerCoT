import copy as cp
from datetime import timedelta
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from fvcore.common.registry import Registry
import torch
import wandb

from common.io_utils import make_dir
from data import build_dataloader
from model import build_model
from optim import build_optim
from eval import build_eval
import pickle
import json
import yaml

TRAINER_REGISTRY = Registry("trainer")
def append_return(lst,eme):
    l = lst.copy()
    l.append(eme)
    return l


def dump_to_yaml(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            for key, value in item.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
class Tracker():
    def __init__(self, cfg):
        self.reset(cfg)

    def step(self):
        self.epoch += 1

    def reset(self, cfg):
        self.exp_name = f"{Path(cfg.exp_dir).name}"
        self.run_id = wandb.util.generate_id()
        self.epoch = 0

    def state_dict(self):
        return {"run_id": self.run_id, "epoch": self.epoch, "exp_name": self.exp_name}
    
    def load_state_dict(self, state_dict):
        state_dict = cp.deepcopy(state_dict)
        self.run_id = state_dict["run_id"]
        self.epoch = state_dict["epoch"]
        self.exp_name = state_dict["exp_name"]

@TRAINER_REGISTRY.register()
class BaseTrainer():
    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.debug = cfg.debug.flag
        self.epochs_per_eval = cfg.solver.get("epochs_per_eval", None)
        self.epochs_per_save = cfg.solver.get("epochs_per_save", None)
        self.global_step = 0
        
        # Initialize accelerator
        self.exp_tracker = Tracker(cfg)
        # There is bug in logger setting, needs fixing from accelerate side
        self.logger = get_logger(__name__)
        self.mode = cfg.mode

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]

        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.solver.get("gradient_accumulation_steps", 1),
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs
        )

        self.accelerator.init_trackers(
                project_name=cfg.name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                init_kwargs={
                    "wandb": {
                        "name": self.exp_tracker.exp_name,
                        "id": self.exp_tracker.run_id, "resume": True
                    }
                }
            )
        print(OmegaConf.to_yaml(cfg))
            
        keys = ["train", "val", "test"]
        self.data_loaders = {key : build_dataloader(cfg, split=key) for key in keys}
        cfg.model.vocab_size = len(self.data_loaders["train"].dataset.tokenizer.vocab)
        self.logger.info(f"Updating vocab size: {cfg.model.vocab_size}")
        self.model = build_model(cfg)
        self.optimizer, self.scheduler = build_optim(cfg, self.model.get_opt_params(),
                                                     total_steps=len(self.data_loaders["train"]) * cfg.solver.epochs)
        self.evaluator = build_eval(cfg, self.accelerator)

        # Training details
        self.epochs = cfg.solver.epochs
        self.total_steps = len(self.data_loaders["train"]) * cfg.solver.epochs
        self.grad_norm = cfg.solver.get("grad_norm")

        # Load pretrain model weights
        if cfg.get('pretrain_ckpt_path'):
            self.pretrain_ckpt_path = Path(cfg.pretrain_ckpt_path)
            self.load_pretrain()

        # Accelerator preparation
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        for name, loader in self.data_loaders.items():
            self.data_loaders[name] = self.accelerator.prepare(loader)
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        # Check if resuming from previous checkpoint is needed
        # self.ckpt_path = '/home/l/Downloads/TransformerCoT/outputs/2023-06-27-08:34:14/ckpt/best.pth'
        self.ckpt_path = Path(cfg.exp_dir) / "ckpt" / "best.pth"
        if cfg.resume:
            self.resume()


    def forward(self, data_dict):
        if self.model.training:
            return self.model(data_dict)
        else:
            return self.model.generate(data_dict)

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, epoch):
        self.model.train()
        loader = self.data_loaders["train"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=f"[Epoch {epoch + 1}/{self.epochs}]")
        for i, data_dict in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                # forward
                data_dict = self.forward(data_dict)
                loss = data_dict['loss']
                # optimize
                self.backward(loss)
                # calculate evaluator
                metrics = self.evaluator.batch_metrics(data_dict)
                # record
                self.global_step += 1
                log_dict = {'step': self.global_step, 'loss': loss.item()}
                log_dict.update(metrics)
                self.log(log_dict, mode="train")
                pbar.update(1)

    @torch.no_grad()
    def eval_step(self, epoch):
        self.model.eval()
        loader = self.data_loaders["val"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            # data_dict = {k : v for k, v in data_dict.items() if isinstance(v, torch.Tensor) or 
            #                 (isinstance(v, list) and isinstance(v[0], torch.Tensor))}
            data_dict = self.accelerator.gather_for_metrics(data_dict)
            self.evaluator.update(data_dict)
            pbar.update(1)
        is_best, results = self.evaluator.record()
        self.log(results, mode="val")
        self.evaluator.reset()
        return is_best



    def test_step(self):
        self.model.eval()
        with open('tokenizer.pkl','rb') as f:
            tokenizer = pickle.load(f)
        loader = self.data_loaders["test"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        data_list = []
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            decoded_text = tokenizer.decode(data_dict['preds'])

            # Calculate the match between cot and preds
            is_match = [True if append_return(cot,out) == pred else False for cot,out, pred in zip(data_dict['cot'], data_dict['output'], decoded_text)]
            output_is_match= [True if append_return(cot,out)[-1] == pred[-1] else False for cot,out, pred in zip(data_dict['cot'], data_dict['output'], decoded_text)]

            data_dict['preds_text'] = decoded_text
            data_dict['match'] = is_match
            data_dict['output_match'] = output_is_match

            data = [{'input': inp, 'output': out,'cot': append_return(cot,out), 'pot': pre, 'cot_match': match, 'output_match': output_is_match} for inp, out,cot, pre, match, output_is_match in zip(data_dict['input'], data_dict['output'],data_dict['cot'],data_dict['preds_text'], data_dict['match'], data_dict['output_match'])]

            data_list+=data
            self.evaluator.update(data_dict)
            pbar.update(1)
        dump_to_yaml(data_list, 'Visualization.yaml')
        is_best, results = self.evaluator.record()

        self.log(results, mode="test")
        self.evaluator.reset()
        return results

    def run(self):
        if self.mode == "train":
            start_epoch = self.exp_tracker.epoch
            self.global_step = start_epoch * len(self.data_loaders["train"])
            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                self.train_step(epoch)

                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    is_best = self.eval_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] finished eval, is_best: {is_best}")
                else:
                    is_best = False

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    if is_best:
                        self.save("best.pth")
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_{epoch+1}.pth")

        self.test_step()
        self.accelerator.end_training()

    def log(self, results, mode="train"):
        if not self.debug:
            log_dict = {}
            for key, val in results.items():
                log_dict[f"{mode}/{key}"] = val
            if mode == "train":
                lrs = self.scheduler.get_lr()
                for i, lr in enumerate(lrs):
                    log_dict[f"{mode}/lr/group_{i}"] = lr
            self.accelerator.log(log_dict, step=self.global_step)

    def save(self, name):
        make_dir(self.ckpt_path.parent)
        self.save_func(str(self.ckpt_path.parent / name))

    def resume(self):
        if self.ckpt_path.exists():
            print(f"Resuming from {str(self.ckpt_path)}")
            # self.logger.info(f"Resuming from {str(self.ckpt_path)}")
            self.accelerator.load_state(str(self.ckpt_path))
            # self.logger.info(f"Successfully resumed from {self.ckpt_path}")
            print(f"Successfully resumed from {self.ckpt_path}")
        else:
            self.logger.info("training from scratch")
            print("training from scratch")

    def load_pretrain(self):
        self.logger.info(f"Loading pretrained weights from {str(self.pretrain_ckpt_path)}")
        model_weight_path = self.pretrain_ckpt_path / "pytorch_model.bin"
        self.model.load_state_dict(torch.load(str(model_weight_path), map_location="cpu"), strict=False)
        self.logger.info(f"Successfully loaded from {str(self.pretrain_ckpt_path)}")

    def save_func(self, path):
        self.accelerator.save_state(path)


def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)