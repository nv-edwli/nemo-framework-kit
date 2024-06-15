echo "Starting the Evaluation process now. For details, see /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py."
echo ""
echo "Working..."

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    trainer.devices=${DEVICES} \
    model.restore_from_path=${RESTORE_FROM_PATH} \
    model.peft.restore_from_path=${PEFT_RESTORE_FROM_PATH} \
    model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    model.pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL_SIZE} \
    model.megatron_amp_O2=${MEGATRON_AMP_02} \
    model.answer_only_loss=${ANSWER_ONLY_LOSS} \
    model.data.test_ds.file_names=${FILE_NAMES} \
    model.data.test_ds.names=${NAMES} \
    model.data.test_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.data.test_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.data.test_ds.tokens_to_generate=${TOKENS_TO_GENERATE} \
    model.data.test_ds.output_file_path_prefix=${OUTPUT_FILE_PATHS_PREFIX} \
    model.data.test_ds.write_predictions_to_file=${WRITE_PREDICTIONS_TO_FILE} \
    inference.greedy=${GREEDY} 

exit 0