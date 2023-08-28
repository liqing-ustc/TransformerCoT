#!/bin/bash
#SBATCH --job-name=cot           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --qos=level0

launcher=${1:-"basic"} # basic for running on your local machine, slurm for running on a cluster

python run.py -m \
    hydra/launcher=${launcher} \
    dataset=CFQ dataset.subset=mcd1 \
    input_types=input \
    output_types=output \
    model.variant=t5-small \
    solver.optim.args.lr=1e-4,5e-4,1e-3,5e-3 \
    dataloader.batch_size=32 solver.gradient_accumulation_steps=4
