# Script for verifying that run_bart_sum can be invoked from its directory

# Get tiny dataset with cnn_dm format (4 examples for train, val, test)
#wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_tiny.tgz
#tar -xzvf cnn_tiny.tgz
#rm cnn_tiny.tgz

export OUTPUT_DIR_NAME=bart_utest_output
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

echo $OUTPUT_DIR
# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py and testing_utils.py
export PYTHONPATH="../":"${PYTHONPATH}"
python finetune.py \
--data_dir=./finetune_ds_no_overlap/ \
--model_name_or_path=facebook/bart-large-cnn \
--learning_rate=3e-5 \
--train_batch_size=1 \
--eval_batch_size=1 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=20  \
--gpus=1 \
--freeze_encoder \
--freeze_embeds \
--max_source_length=1024 \
--max_target_length=400 \
--val_max_target_length=400 \
--test_max_target_length=400 \
--encoder_layerdrop 0.1 \
--early_stopping_patience 3 \
--do_train "$@" \
--logger_name wandb \
--val_metric=rouge1 \
--eval_max_gen_length=400 \

#rm -rf cnn_tiny
#rm -rf $OUTPUT_DIR



