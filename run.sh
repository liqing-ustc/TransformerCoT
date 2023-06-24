#/usr/bin/env bash
python run.py -m \
    rng_seed=1,2,3 \
    use_cot=True \
    dataset.subset=length \
    model.position_embedding_type=relative_key \
    model.distance_clip=10 \
    solver.epochs_per_eval=10 solver.epochs=100