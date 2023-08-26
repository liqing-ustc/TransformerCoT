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

python run.py -m \
    input_types=rir,output \
    output_types=output \
    model.variant=t5-small dataloader.batch_size=32

# python run.py -m \
#     rng_seed=1,2,3 \
#     use_cot=True \
#     dataset.subset=length \
#     model.position_embedding_type=relative_key \
#     model.distance_clip=10 \
#     solver.epochs_per_eval=10 solver.epochs=100