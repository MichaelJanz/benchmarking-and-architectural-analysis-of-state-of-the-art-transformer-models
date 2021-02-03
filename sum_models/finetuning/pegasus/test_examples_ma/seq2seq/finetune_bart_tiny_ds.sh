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
--data_dir=./finetune_ds_overlap/ \
--model_name_or_path=facebook/bart-large-cnn  \
--learning_rate=3e-5 \
--freeze_encoder \
--freeze_embeds \
--train_batch_size=1 \
--eval_batch_size=1 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=10  \
--gpus=1 \
--logger_name=wandb \
--max_target_length=500 \
--val_max_target_length=500 \
--test_max_target_length=500 \
--encoder_layerdrop=0.2 \
--early_stopping_patience 10 \
--do_train "$@" \
--val_metric=rouge2
#--fp16 \

#rm -rf cnn_tiny
#rm -rf $OUTPUT_DIR



