#!/bin/bash

BERT_MODEL=bert-large-uncased
BATCH_SIZE=16
NUM_EPOCHS=3
SAVE_STEPS=750
REINIT_LAYERS=1

for DS_ID in 1 2 3 4
do
	DATA_DIR="data/dataset"$DS_ID
	for SEED in 1 2 3
	do
		OUTPUT_DIR="output/dataset"$DS_ID"/seed"$SEED
		poetry run python3 music-ner/src/fine-tune.py --dataset_name music-ner/datasets --model_name_or_path $BERT_MODEL --output_dir $OUTPUT_DIR --num_train_epochs $NUM_EPOCHS --per_device_train_batch_size $BATCH_SIZE --seed $SEED --do_train --do_predict --overwrite_output_dir  --reinit_layers $REINIT_LAYERS --return_entity_level_metrics --dataset_path=$DATA_DIR
	done
done
