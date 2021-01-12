# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
--data_dir=./finetune_ds \
--learning_rate=3e-5 \
--train_batch_size=2 \
--eval_batch_size=2 \
--output_dir=./gs_results \
--max_source_length=512 \
--max_target_length=56 \
--val_check_interval=0.1 --n_val=200 \
--do_train --do_predict \
 "$@"
