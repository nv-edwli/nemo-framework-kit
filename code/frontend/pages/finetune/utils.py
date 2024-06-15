# Helper functions for the application

import os, sys, yaml

def checkpoint_exists(model: str):
    if model == "Llama 2 7B":
        return os.path.exists("/project/models/llama2-7b.nemo")
    elif model == "Llama 2 13B":
        return os.path.exists("/project/models/llama2-13b.nemo")
    elif model == "Llama 2 70B":
        return os.path.exists("/project/models/llama2-70b.nemo")
    return None

def get_parameter_count(model: str):
    if model == "Llama 2 7B":
        return "7b"
    elif model == "Llama 2 13B":
        return "13b"
    elif model == "Llama 2 70B":
        return "70b"
    return None

def retrieve_configs():
    with open("/project/code/configs/config-default.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return {
        "log_every_n_steps": config["trainer"]["log_every_n_steps"], 
        "precision": config["trainer"]["precision"], 
        "devices": config["trainer"]["devices"], 
        "num_nodes": config["trainer"]["num_nodes"], 
        "val_check_interval": config["trainer"]["val_check_interval"], 
        "max_steps": config["trainer"]["max_steps"], 
        "restore_from_path": config["model"]["restore_from_path"], 
        "peft_scheme": config["model"]["peft"]["peft_scheme"], 
        "micro_batch_size": config["model"]["micro_batch_size"], 
        "global_batch_size": config["model"]["global_batch_size"], 
        "tensor_model_parallel_size": config["model"]["tensor_model_parallel_size"], 
        "pipeline_model_parallel_size": config["model"]["pipeline_model_parallel_size"], 
        "megatron_amp_O2": config["model"]["megatron_amp_O2"], 
        "activations_checkpoint_granularity": config["model"]["activations_checkpoint_granularity"], 
        "activations_checkpoint_num_layers": config["model"]["activations_checkpoint_num_layers"], 
        "activations_checkpoint_method": config["model"]["activations_checkpoint_method"], 
        "optim_name": config["model"]["optim"]["name"], 
        "lr": config["model"]["optim"]["lr"], 
        "answer_only_loss": config["model"]["answer_only_loss"], 
        "train_file_names": config["model"]["data"]["train_ds"]["file_names"], 
        "train_concat_sampling_probabilities": config["model"]["data"]["train_ds"]["concat_sampling_probabilities"], 
        "train_max_seq_length": config["model"]["data"]["train_ds"]["max_seq_length"], 
        "train_micro_batch_size": config["model"]["data"]["train_ds"]["micro_batch_size"], 
        "train_global_batch_size": config["model"]["data"]["train_ds"]["global_batch_size"], 
        "train_num_workers": config["model"]["data"]["train_ds"]["num_workers"], 
        "val_file_names": config["model"]["data"]["validation_ds"]["file_names"], 
        "val_max_seq_length": config["model"]["data"]["validation_ds"]["max_seq_length"], 
        "val_micro_batch_size": config["model"]["data"]["validation_ds"]["micro_batch_size"], 
        "val_global_batch_size": config["model"]["data"]["validation_ds"]["global_batch_size"], 
        "val_num_workers": config["model"]["data"]["validation_ds"]["num_workers"], 
        "val_name": config["model"]["data"]["validation_ds"]["metric"]["name"], 
        "test_file_names": config["model"]["data"]["test_ds"]["file_names"], 
        "test_num_workers": config["model"]["data"]["test_ds"]["num_workers"], 
        "test_name": config["model"]["data"]["test_ds"]["metric"]["name"], 
        "save_nemo_on_validation_end": config["model"]["save_nemo_on_validation_end"], 
        "create_wandb_logger": config["exp_manager"]["create_wandb_logger"],             
        "checkpoint_callback_params_mode": config["exp_manager"]["checkpoint_callback_params_mode"],             
        "explicit_log_dir": config["exp_manager"]["explicit_log_dir"],             
        "resume_if_exists": config["exp_manager"]["resume_if_exists"],             
        "resume_ignore_no_checkpoint": config["exp_manager"]["resume_ignore_no_checkpoint"],       
        "create_checkpoint_callback": config["exp_manager"]["create_checkpoint_callback"],       
        "monitor": config["exp_manager"]["checkpoint_callback_params"]["monitor"],        
        "save_best_model": config["exp_manager"]["checkpoint_callback_params"]["save_best_model"],        
        "save_nemo_on_train_end": config["exp_manager"]["checkpoint_callback_params"]["save_nemo_on_train_end"]
    }

def retrieve_eval_configs():
    with open("/project/code/configs/config-eval.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return {
        "devices": config["trainer"]["devices"], 
        "restore_from_path": config["model"]["restore_from_path"], 
        "peft_restore_from_path": config["model"]["peft"]["restore_from_path"], 
        "tensor_model_parallel_size": config["model"]["tensor_model_parallel_size"], 
        "pipeline_model_parallel_size": config["model"]["pipeline_model_parallel_size"], 
        "megatron_amp_O2": config["model"]["megatron_amp_O2"], 
        "answer_only_loss": config["model"]["answer_only_loss"], 
        "file_names": config["model"]["data"]["test_ds"]["file_names"], 
        "names": config["model"]["data"]["test_ds"]["names"], 
        "micro_batch_size": config["model"]["data"]["test_ds"]["micro_batch_size"], 
        "global_batch_size": config["model"]["data"]["test_ds"]["global_batch_size"], 
        "tokens_to_generate": config["model"]["data"]["test_ds"]["tokens_to_generate"], 
        "output_file_path_prefix": config["model"]["data"]["test_ds"]["output_file_path_prefix"], 
        "write_predictions_to_file": config["model"]["data"]["test_ds"]["write_predictions_to_file"], 
        "greedy": config["inference"]["greedy"]
    }

def clear_active_data_dir(folder):
    for filename in os.listdir(folder):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def parse_lang(file):
    if file[-2:] == "py":
        return "python"
    elif file[-2:] == "sh" or file[-4:] == "bash":
        return "shell"
    elif file[-2:] == "md":
        return "markdown"
    elif file[-4:] == "yaml" or file[-3:] == "yml":
        return "yaml"
    elif file[-4:] == "json":
        return "json"
    elif file[-4:] == "html":
        return "html"
    elif file[-3:] == "css":
        return "css"
    elif file[-2:] == "js":
        return "javascript"
    else:
        return None
    
def read_logs():
    sys.stdout.flush()
    with open("/project/models/output.log", "r") as f:
        return f.read()