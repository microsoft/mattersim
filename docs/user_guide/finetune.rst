
Finetune MatterSim
==================

Finetune Script
---------------

MatterSim provides a finetune script to
finetune the pre-trained MatterSim model on a custom dataset.
You can find the script in the ``training`` folder or in the
`github link <https://github.com/microsoft/mattersim/blob/main/src/mattersim/training/finetune_mattersim.py>`_.

Finetune Parameters
--------------------

The finetune script accepts several command-line arguments to customize the training process. Below is a list of the available parameters:

- **run_name**: (str) The name of the run. Default is "example".

- **train_data_path**: (str) Path to the training data file. Supports various file types readable by ASE (e.g., `.xyz`, `.traj`, `.cif`) and `.pkl` files. Default is "./sample.xyz".

- **valid_data_path**: (str) Path to the validation data file. Default is None.

- **load_model_path**: (str) Path to load the pre-trained model. Default is "mattersim-v1.0.0-1m".

- **save_path**: (str) Path to save the trained model. Default is "./results".

- **save_checkpoint**: (bool) Whether to save checkpoints during training. Default is False.

- **ckpt_interval**: (int) Interval (in epochs) to save checkpoints. Default is 10.

- **device**: (str) Device to use for training, either "cuda" or "cpu". Default is "cuda".

- **cutoff**: (float) Cutoff radius for interactions. Default is 5.0.

- **threebody_cutoff**: (float) Cutoff radius for three-body interactions, should be smaller than the two-body cutoff. Default is 4.0.

- **epochs**: (int) Number of training epochs. Default is 1000.

- **batch_size**: (int) Batch size for training. Default is 16.

- **lr**: (float) Learning rate for the optimizer. Default is 2e-4.

- **step_size**: (int) Step size for the learning rate scheduler. Default is 10.

- **include_forces**: (bool) Whether to include forces in the training. Default is True.

- **include_stresses**: (bool) Whether to include stresses in the training. Default is False.

- **force_loss_ratio**: (float) Ratio of force loss in the total loss. Default is 1.0.

- **stress_loss_ratio**: (float) Ratio of stress loss in the total loss. Default is 0.1.

- **early_stop_patience**: (int) Patience for early stopping. Default is 10.

- **seed**: (int) Random seed for reproducibility. Default is 42.

- **re_normalize**: (bool) Whether to re-normalize energy and forces according to new data. Default is False.

- **scale_key**: (str) Key for scaling forces. Only used when ``re_normalize`` is True. Default is "per_species_forces_rms".

- **shift_key**: (str) Key for shifting energy. Only used when ``re_normalize`` is True. Default is "per_species_energy_mean_linear_reg".

- **init_scale**: (float) Initial scale value. Only used when ``re_normalize`` is True. Default is None.

- **init_shift**: (float) Initial shift value. Only used when ``re_normalize`` is True. Default is None.

- **trainable_scale**: (bool) Whether the scale is trainable. Only used when ``re_normalize`` is True. Default is False.

- **trainable_shift**: (bool) Whether the shift is trainable. Only used when ``re_normalize`` is True. Default is False.

- **wandb**: (bool) Whether to use Weights & Biases for logging. Default is False.

- **wandb_api_key**: (str) API key for Weights & Biases. Default is None.

- **wandb_project**: (str) Project name for Weights & Biases. Default is "wandb_test".

These parameters allow you to customize the finetuning process to suit your specific dataset and computational resources.

Finetune Example
----------------
You can replace the data path with your own data path.

.. code-block:: bash

    torchrun --nproc_per_node=1 src/mattersim/training/finetune_mattersim.py --load_model_path mattersim-v1.0.0-1m --train_data_path xyz_files/train.xyz --valid_data_path xyz_files/valid.xyz --batch_size 16 --lr 2e-4 --step_size 20 --epochs 200 --save_path ./finetune_result --save_checkpoint --ckpt_interval 20 --include_stresses --include_forces
