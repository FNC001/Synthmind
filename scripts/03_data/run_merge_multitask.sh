TASK_ROOTS=$(find /Users/wyc/MP_exp_doi/data/interim/training_modes \
  -maxdepth 1 -type d -name 'stage3_*' | sort)

python merge_stage3_task_modes_to_multitask_table.py \
  --task_mode_roots $TASK_ROOTS \
  --train_mode relaxed_only \
  --output_dir /Users/wyc/MP_exp_doi/data/interim/merged_stage3_multitask/relaxed_only

TASK_ROOTS=$(find /Users/wyc/MP_exp_doi/data/interim/training_modes \
  -maxdepth 1 -type d -name 'stage3_*' | sort)

echo "$TASK_ROOTS"

python merge_stage3_task_modes_to_multitask_table.py \
  --task_mode_roots $TASK_ROOTS \
  --train_mode relaxed_only \
  --output_dir /Users/wyc/MP_exp_doi/data/interim/merged_stage3_multitask/relaxed_only
