echo "Starting the Finetuning process now. For details, see /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py."
echo ""
echo "Working..."

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Run the PEFT command by appropriately setting the values for the parameters such as the number of steps,
# model checkpoint path, batch sizes etc. For a full reference of parameter
# settings refer to the config at https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
    trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
    trainer.precision=${PRECISION} \
    trainer.devices=${DEVICES} \
    trainer.num_nodes=${NUM_NODES} \
    trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    trainer.max_steps=${MAX_STEPS} \
    model.restore_from_path=${RESTORE_FROM_PATH} \
    model.peft.peft_scheme=${PEFT_SCHEME} \
    model.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    model.pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL_SIZE} \
    model.megatron_amp_O2=${MEGATRON_AMP_02} \
    model.activations_checkpoint_granularity=${ACTIVATIONS_CHECKPOINT_GRANULARITY} \
    model.activations_checkpoint_num_layers=${ACTIVATIONS_CHECKPOINT_NUM_LAYERS} \
    model.activations_checkpoint_method=${ACTIVATIONS_CHECKPOINT_METHOD} \
    model.optim.name=${OPTIM_NAME} \
    model.optim.lr=${LR} \
    model.answer_only_loss=${ANSWER_ONLY_LOSS} \
    model.data.train_ds.file_names=${TRAIN_FILE_NAMES} \
    model.data.train_ds.concat_sampling_probabilities=${TRAIN_CONCAT_SAMPLING_PROBABILITIES} \
    model.data.train_ds.max_seq_length=${TRAIN_MAX_SEQ_LENGTH} \
    model.data.train_ds.micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
    model.data.train_ds.global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
    model.data.train_ds.num_workers=${TRAIN_NUM_WORKERS} \
    model.data.validation_ds.file_names=${VAL_FILE_NAMES} \
    model.data.validation_ds.max_seq_length=${VAL_MAX_SEQ_LENGTH} \
    model.data.validation_ds.micro_batch_size=${VAL_MICRO_BATCH_SIZE} \
    model.data.validation_ds.global_batch_size=${VAL_GLOBAL_BATCH_SIZE} \
    model.data.validation_ds.num_workers=${VAL_NUM_WORKERS} \
    model.data.validation_ds.metric.name=${VAL_NAME} \
    model.data.test_ds.file_names=${TEST_FILE_NAMES} \
    model.data.test_ds.num_workers=${TEST_NUM_WORKERS} \
    model.data.test_ds.metric.name=${TEST_NAME} \
    model.save_nemo_on_validation_end=${SAVE_NEMO_ON_VALIDATION_END} \
    exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER} \
    exp_manager.checkpoint_callback_params.mode=${CHECKPOINT_CALLBACK_PARAMS_MODE} \
    exp_manager.explicit_log_dir=${EXPLICIT_LOG_DIR} \
    exp_manager.resume_if_exists=${RESUME_IF_EXISTS} \
    exp_manager.resume_ignore_no_checkpoint=${RESUME_IGNORE_NO_CHECKPOINT} \
    exp_manager.create_checkpoint_callback=${CREATE_CHECKPOINT_CALLBACK} \
    exp_manager.checkpoint_callback_params.monitor=${MONITOR} \
    ++exp_manager.checkpoint_callback_params.save_best_model=${SAVE_BEST_MODEL} \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=${SAVE_NEMO_ON_TRAIN_END} \

exit 0