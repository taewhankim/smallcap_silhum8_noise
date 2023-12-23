#!/bin/bash

MODEL_PATH=$1

#li=("checkpoint-8856" "checkpoint-13284" "checkpoint-17712" "checkpoint-22140" "checkpoint-26568" "checkpoint-30996" "checkpoint-35424" "checkpoint-39852" "checkpoint-4428" "checkpoint-44280" "checkpoint-8856")
li=("checkpoint-4428" "checkpoint-8856" "checkpoint-13284" "checkpoint-17712" "checkpoint-22140" "checkpoint-26568" "checkpoint-30996" "checkpoint-35424" "checkpoint-39852" "checkpoint-44280")

#declare -a li=("checkpoint-26568" "checkpoint-30996" "checkpoint-35424" "checkpoint-39852" "checkpoint-4428" "checkpoint-44280" "checkpoint-8856");


for l in "${li[@]}";
do
    echo "$l @@@@@@@@@@@@@@@@@@@"
    python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "$MODEL_PATH"/"$l"/test_preds.json
done

