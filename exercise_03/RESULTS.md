# Exercise 03 status

Implemented the required transforms, ImageFolder dataset, PlantCNN, accuracy
calculation, training and validation loops, TensorBoard logging, and checkpointing.
The comparison configuration is `config_no_augmentation.yaml`.

## Verification

- All 32 supplied checks passed on CPU using temporary image fixtures.
- A small CPU integration check verified parameter updates, validation without
  changing model state, checkpoint reload, and test evaluation metrics.
- These checks verify the implementation; they do not measure PlantWild accuracy.

## Training and evaluation remaining

The documented shared dataset path
`/ai2_ex/data/classification/plantwild_subset` returns `Permission denied` for this
account. Full training, augmentation comparison, and held-out test evaluation
have not run. An accessible dataset path or corrected permissions are required.

Once access is available, run from `exercise_03/`:

```bash
ln -s /ai2_ex/data/classification/plantwild_subset data/plantwild
```

Check the class count and set `model.num_classes` in both configurations to
match. Submit the first experiment:

```bash
sbatch submit_slurm.sh config.yaml
```

After it finishes, submit the comparison (only one job should run at a time):

```bash
sbatch submit_slurm.sh config_no_augmentation.yaml
```

Compare validation curves with TensorBoard:

```bash
tensorboard --logdir outputs/logs --port 6006
```

After training, evaluate each best checkpoint, one job at a time:

```bash
sbatch submit_slurm_eval.sh config.yaml outputs/checkpoints/plantwild_baseline_best.pt
sbatch submit_slurm_eval.sh config_no_augmentation.yaml outputs/checkpoints/plantwild_no_augmentation_best.pt
```

Record real validation/test accuracy and the augmentation comparison here after
those jobs complete. No accuracy results are claimed yet.
