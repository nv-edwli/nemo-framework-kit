# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A finetuning dashboard to customize a base model."""

from pathlib import Path

import gradio as gr
import yaml

# from . import mermaid
from ...common import IMG_DIR, THEME, USE_KB_INITIAL
from ...configuration import config

import os
import subprocess
import yaml
import shutil
import time 
import sys
from . import markdown, utils

# load custom style and scripts
_CSS_FILE = Path(__file__).parent.joinpath("style.css")
_CSS = open(_CSS_FILE, "r", encoding="UTF-8").read()
_MMD = [
    """flowchart LR
    query(fa:fa-user Query) -->
    prompt(LLM Context) -->
    llm(LLM NIM):::nvidia -->
    answer(fa:fa-comment-dots Answer)

    classDef nvidia fill:#76b900,stroke:#333,stroke-width:1px;""",
    """flowchart LR
    ret <-....-> db[(fa:fa-building\nEnterprise\nData)]

    query(fa:fa-user\nQuery) --> prompt
    query -->
    ret(Retrieval NIM):::nvidia -->
    prompt(LLM Context) -->
    llm(LLM NIM):::nvidia -->
    answer(fa:fa-comment-dots\nAnswer)

    classDef nvidia fill:#76b900,stroke:#333,stroke-width:1px;""",
]

_KB_TOGGLE_JS = """
async(val) => {
    window.top.postMessage({"use_kb": val}, '*');
}
"""
_CONFIG_CHANGES_JS = """
async() => {
    title = document.querySelector("div#config-toolbar p");
    if (! title.innerHTML.endsWith("🟠")) { title.innerHTML = title.innerHTML.slice(0,-2) + "🟠"; };
}
"""
_SAVE_CHANGES_JS = """
async() => {
    title = document.querySelector("div#config-toolbar p");
    if (! title.innerHTML.endsWith("🟢")) { title.innerHTML = title.innerHTML.slice(0,-2) + "🟢"; };
}
"""

_SAVE_IMG = IMG_DIR.joinpath("floppy.svg")
_UNDO_IMG = IMG_DIR.joinpath("undo.svg")
_HISTORY_IMG = IMG_DIR.joinpath("history.svg")
_PSEUDO_FILE_NAME = "config-default.yaml 🟢"
_PSEUDO_FILE_EVAL_NAME = "config-eval.yaml 🟢"
_STARTING_CONFIG = open("/project/code/configs/config-default.yaml", "r", encoding="UTF-8").read()
_STARTING_EVAL_CONFIG = open("/project/code/configs/config-eval.yaml", "r", encoding="UTF-8").read()

# web ui definition
with gr.Blocks(theme=THEME, css=_CSS) as page:
    
    selected_model = gr.State("/project/models/llama2-7b.nemo" if utils.checkpoint_exists("Llama 2 7B") else "")
    selected_train_ds = gr.State("")
    selected_val_ds = gr.State("")
    selected_test_ds = gr.State("")
    
    with gr.Tab("Code", elem_id="cp-tab", elem_classes=["invert-bg"]):
        with gr.Row(elem_id="config-row"):
            gr.Markdown(markdown.CODE_MARKDOWN)
        with gr.Row(elem_id="config-row"):
            with gr.Column(scale=1):
                filesystem = gr.FileExplorer(
                    glob="/*",
                    file_count="single",
                    ignore_glob="/project/.*",
                    root="/project",
                    label="/project/"
                )
            with gr.Column(scale=3):
                selected_code = gr.Code(show_label=False)
            
        def _get_file_content(file):
            if file is not None and not os.path.isdir(file):
                return {
                    selected_code: gr.update(value=(file,), language=utils.parse_lang(file))
                }
            else:
                return {
                    selected_code: gr.update(value=None)
                }
        
        filesystem.change(_get_file_content, [filesystem], [selected_code])
    
    with gr.Tab("Model", elem_id="cp-tab", elem_classes=["invert-bg"]):
        with gr.Row(elem_id="config-row"):
            gr.Markdown(markdown.MODEL_MARKDOWN)
        with gr.Row(elem_id="config-row"):
            with gr.Column():
                with gr.Row(elem_id="config-row"):
                    select_model = gr.Dropdown(["Llama 2 7B", "Llama 2 13B", "Llama 2 70B"], value="Llama 2 7B", label="Select a Model", scale=2)
                    load_model = gr.Button("Download NeMo Checkpoint", scale=1, 
                                              variant="primary" if utils.checkpoint_exists("Llama 2 7B") else "secondary",
                                              interactive=False if utils.checkpoint_exists("Llama 2 7B") else True)
                
        def _select_model(select_model: str):
            if utils.checkpoint_exists(select_model) is not None: 
                if "70B" in select_model:
                    updated_model = "/project/models/llama2-70b.nemo"
                elif "13B" in select_model:
                    updated_model = "/project/models/llama2-13b.nemo"
                else:
                    updated_model = "/project/models/llama2-7b.nemo"
                return {
                    load_model: gr.update(variant="primary" if utils.checkpoint_exists(select_model) else "secondary",
                                          interactive=False if utils.checkpoint_exists(select_model) else True),
                    selected_model: updated_model if utils.checkpoint_exists(select_model) else ""
                }
            else:
                gr.Warning("Model Invalid.")
                return {
                    load_model: gr.update(variant="secondary", interactive=True), 
                    selected_model: ""
                }
            
        select_model.change(_select_model, [select_model], [load_model, selected_model])
                                           
        def _load_model(select_model: str):
            rc = subprocess.call("/bin/bash /project/code/scripts/init.sh " + utils.get_parameter_count(select_model) + " >> " + "/project/models/output.log",
                                 shell=True)
            if rc == 0:
                return {
                    load_model: gr.update(value="NeMo Checkpoint Ready", variant="primary", interactive=False), 
                    selected_model: "/project/models/llama2-7b.nemo" if select_model == "Llama 2 7B" else 
                    ("/project/models/llama2-13b.nemo" if select_model == "Llama 2 13B" else "/project/models/llama2-70b.nemo")
                }
            else:
                gr.Warning("Unable to download and/or convert model weights to NeMo format.")
                return {
                    load_model: gr.update(variant="secondary", interactive=True), 
                    selected_model: ""
                }
            
        load_model.click(_load_model, [select_model], [load_model, selected_model])

    with gr.Tab("Dataset", elem_classes=["invert-bg"]):
        with gr.Row(elem_id="config-row"):
            gr.Markdown(markdown.DATA_MARKDOWN)
        with gr.Row(elem_id="config-row"):
            with gr.Column():
                with gr.Row(elem_id="config-row") as dataset_1:
                    with gr.Column(scale=1):
                        gr.Markdown("Which dataset would you like to use?")
                    with gr.Column(scale=1):
                        with gr.Row(elem_id="config-row"):
                            preload_data = gr.Button("Use a pre-loaded dataset", size="sm", scale=1)
                            bring_data = gr.Button("Bring my own dataset", size="sm", scale=1)
                gr.Markdown("<br />")
                with gr.Row(elem_id="config-row", visible=False) as dataset_2:
                    select_dataset = gr.Dropdown(choices=["Select", "SQuAD", "PubMedQA"], value="Select", label="Select a Dataset", interactive=True, scale=3)
                gr.Markdown("<br />")
                with gr.Row(elem_id="config-row", visible=False) as preloaded:
                    with gr.Column(visible=False) as pubmed:
                        with gr.Row(elem_id="config-row"):
                            gr.File(value="/project/data/default_data/PubMedQA/pubmedqa_train.jsonl", label="TRAINING dataset", interactive=False)
                        with gr.Row(elem_id="config-row"):
                            gr.File(value="/project/data/default_data/PubMedQA/pubmedqa_val.jsonl", label="VALIDATION dataset", interactive=False)
                        with gr.Row(elem_id="config-row"):
                            gr.File(value="/project/data/default_data/PubMedQA/pubmedqa_test.jsonl", label="TEST dataset", interactive=False)
                    with gr.Column(visible=False) as squad:
                        with gr.Row(elem_id="config-row"):
                            gr.File(value="/project/data/default_data/SQuAD/squad_short_train.jsonl", label="TRAINING dataset", interactive=False)
                        with gr.Row(elem_id="config-row"):
                            gr.File(value="/project/data/default_data/SQuAD/squad_short_val.jsonl", label="VALIDATION dataset", interactive=False)
                        with gr.Row(elem_id="config-row"):
                            gr.File(value="/project/data/default_data/SQuAD/squad_test.jsonl", label="TEST dataset", interactive=False)
                with gr.Row(elem_id="config-row", visible=False) as dataset_3:
                    with gr.Column():
                        with gr.Row(elem_id="config-row"):
                            train_ds = gr.File(file_types=[".jsonl"], label="TRAINING dataset (.jsonl)")
                        with gr.Row(elem_id="config-row"):
                            val_ds = gr.File(file_types=[".jsonl"], label="VALIDATION dataset (.jsonl)")
                        with gr.Row(elem_id="config-row"):
                            test_ds = gr.File(file_types=[".jsonl"], label="TEST dataset (.jsonl)")
                
                
        def _dataset_fork(btn: str, curr_train_path: str, curr_val_path: str, curr_test_path: str):
            utils.clear_active_data_dir("/project/data/active_data") # Remove any existing files from the active_data dir
            if btn == "Use a pre-loaded dataset":
                return {
                    preload_data: gr.update(variant="primary", interactive=False),
                    bring_data: gr.update(variant="secondary", interactive=True),
                    select_dataset: gr.update(visible=True),
                    dataset_2: gr.update(visible=True),
                    dataset_3: gr.update(visible=False),
                    train_ds: gr.update(value=None),
                    val_ds: gr.update(value=None),
                    test_ds: gr.update(value=None),
                    selected_train_ds: "",
                    selected_val_ds: "",
                    selected_test_ds: "",
                    preloaded: gr.update(visible=False),
                }
            elif btn == "Bring my own dataset":
                return {
                    preload_data: gr.update(variant="secondary", interactive=True),
                    bring_data: gr.update(variant="primary", interactive=False),
                    select_dataset: gr.update(value="Select"),
                    dataset_2: gr.update(visible=False),
                    dataset_3: gr.update(visible=True),
                    train_ds: gr.update(value=None),
                    val_ds: gr.update(value=None),
                    test_ds: gr.update(value=None),
                    selected_train_ds: "",
                    selected_val_ds: "",
                    selected_test_ds: "",
                    preloaded: gr.update(visible=False),
                }
            return {
                preload_data: gr.update(variant="secondary", interactive=True),
                bring_data: gr.update(variant="secondary", interactive=True),
                select_dataset: gr.update(value="Select"),
                dataset_2: gr.update(visible=False),
                dataset_3: gr.update(visible=False),
                train_ds: gr.update(visible=True),
                val_ds: gr.update(visible=True),
                test_ds: gr.update(visible=True),
                selected_train_ds: "",
                selected_val_ds: "",
                selected_test_ds: "",
                preloaded: gr.update(elem_id="config-row"),
            }
        
        preload_data.click(_dataset_fork, [preload_data, selected_train_ds, selected_val_ds, selected_test_ds], [preload_data, 
                                                                                                                   bring_data, 
                                                                                                                   select_dataset, 
                                                                                                                   dataset_2, 
                                                                                                                   dataset_3,
                                                                                                                   train_ds,
                                                                                                                   val_ds,
                                                                                                                   test_ds,
                                                                                                                   selected_train_ds,
                                                                                                                   selected_val_ds,
                                                                                                                   selected_test_ds,
                                                                                                                   preloaded])
        bring_data.click(_dataset_fork, [bring_data, selected_train_ds, selected_val_ds, selected_test_ds], [preload_data, 
                                                                                                               bring_data, 
                                                                                                               select_dataset, 
                                                                                                               dataset_2, 
                                                                                                               dataset_3,
                                                                                                               train_ds,
                                                                                                               val_ds,
                                                                                                               test_ds,
                                                                                                               selected_train_ds,
                                                                                                               selected_val_ds,
                                                                                                               selected_test_ds,
                                                                                                               preloaded])

        def _load_dataset(dataset):
            if dataset == "SQuAD":
                train_path = "/project/data/default_data/" + dataset + "/squad_short_train.jsonl" 
                val_path = "/project/data/default_data/" + dataset + "/squad_short_val.jsonl" 
                test_path = "/project/data/default_data/" + dataset + "/squad_test.jsonl"
                shutil.copyfile(train_path, "/project/data/active_data/squad_short_train.jsonl")
                shutil.copyfile(val_path, "/project/data/active_data/squad_short_val.jsonl")
                shutil.copyfile(test_path, "/project/data/active_data/squad_test.jsonl")
                return {
                    selected_train_ds: "/project/data/active_data/squad_short_train.jsonl",
                    selected_val_ds: "/project/data/active_data/squad_short_val.jsonl",
                    selected_test_ds: "/project/data/active_data/squad_test.jsonl",
                    preloaded: gr.update(visible=True),
                    squad: gr.update(visible=True),
                    pubmed: gr.update(visible=False),
                }
            elif dataset == "PubMedQA":
                train_path = "/project/data/default_data/" + dataset + "/pubmedqa_train.jsonl" 
                val_path = "/project/data/default_data/" + dataset + "/pubmedqa_val.jsonl" 
                test_path = "/project/data/default_data/" + dataset + "/pubmedqa_test.jsonl"
                shutil.copyfile(train_path, "/project/data/active_data/pubmedqa_train.jsonl")
                shutil.copyfile(val_path, "/project/data/active_data/pubmedqa_val.jsonl")
                shutil.copyfile(test_path, "/project/data/active_data/pubmedqa_test.jsonl")
                return {
                    selected_train_ds: "/project/data/active_data/pubmedqa_train.jsonl",
                    selected_val_ds: "/project/data/active_data/pubmedqa_val.jsonl",
                    selected_test_ds: "/project/data/active_data/pubmedqa_test.jsonl",
                    preloaded: gr.update(visible=True),
                    squad: gr.update(visible=False),
                    pubmed: gr.update(visible=True),
                }
            return {
                selected_train_ds: "",
                selected_val_ds: "",
                selected_test_ds: "",
                preloaded: gr.update(visible=False),
                squad: gr.update(visible=False),
                pubmed: gr.update(visible=False),
            }

        select_dataset.change(_load_dataset, [select_dataset], [selected_train_ds, selected_val_ds, selected_test_ds, preloaded, squad, pubmed])
        
        def _process_train(fileobj):
            # Copy file contents to location in filesystem for fine-tuning script to use.
            path = "/project/data/active_data/" + os.path.basename(fileobj)  #NB*
            shutil.copyfile(fileobj.name, path)
            return path
        
        def _process_val(fileobj):
            # Copy file contents to location in filesystem for fine-tuning script to use.
            path = "/project/data/active_data/" + os.path.basename(fileobj)  #NB*
            shutil.copyfile(fileobj.name, path)
            return path
        
        def _process_test(fileobj):
            # Copy file contents to location in filesystem for fine-tuning script to use.
            path = "/project/data/active_data/" + os.path.basename(fileobj)  #NB*
            shutil.copyfile(fileobj.name, path)
            return path
        
        train_ds.upload(_process_train, [train_ds], [selected_train_ds])
        val_ds.upload(_process_val, [val_ds], [selected_val_ds])
        test_ds.upload(_process_test, [test_ds], [selected_test_ds])
        
        def _clear_train(selected_train_ds):
            try:
                os.remove(selected_train_ds)
            except OSError:
                pass
            return ""
        def _clear_val(selected_val_ds):
            try:
                os.remove(selected_val_ds)
            except OSError:
                pass
            return ""
        def _clear_test(selected_test_ds):
            try:
                os.remove(selected_test_ds)
            except OSError:
                pass
            return ""
        
        train_ds.clear(_clear_train, [selected_train_ds], [selected_train_ds])
        val_ds.clear(_clear_val, [selected_val_ds], [selected_val_ds])
        test_ds.clear(_clear_test, [selected_test_ds], [selected_test_ds])
    
    with gr.Tab("Finetune", elem_classes=["invert-bg"]) as ft_tab:
        with gr.Row(elem_id="config-row", visible=True) as ft_not_ready:
            err_msg = gr.Markdown("You are not ready to start finetuning: Load in the models and select a dataset.")
        with gr.Row(elem_id="config-row", visible=False) as ft_ready:
            with gr.Column():
                with gr.Row(elem_id="config-row"):
                    with gr.Column(scale=1):
                        gr.Markdown(markdown.CONFIG_MARKDOWN)
                    with gr.Column(scale=1):
                        with gr.Row(elem_id="config-row"):
                            gr.Markdown(markdown.FINETUNE_MARKDOWN)
                        with gr.Row(elem_id="config-row"):
                            ft_ack = gr.Checkbox(label="I have finished the below configuration file.")
                        with gr.Row(elem_id="config-row", visible=False) as ft_start_row:
                            ft_start = gr.Button("Start Fine-tuning", size="sm", scale=1)
                with gr.Row(elem_id="config-row"):
                    with gr.Group(elem_id="config-wrapper"):
                        with gr.Row(elem_id="config-toolbar", elem_classes=["toolbar"]):
                            file_title = gr.Markdown(_PSEUDO_FILE_NAME, elem_id="editor-title")
                            save_btn = gr.Button("", icon=_SAVE_IMG, elem_classes=["toolbar"])
                            undo_btn = gr.Button("", icon=_UNDO_IMG, elem_classes=["toolbar"])
                            reset_btn = gr.Button("", icon=_HISTORY_IMG, elem_classes=["toolbar"])
                        with gr.Row(elem_id="config-row-box"):
                            editor = gr.Code(
                                _STARTING_CONFIG,
                                elem_id="config-editor",
                                interactive=True,
                                language="yaml",
                                show_label=False,
                                container=False,
                            )
                            
        def _ack_ft_cfg(ft_ack):
            return {
                ft_start_row: gr.update(visible=ft_ack)
            }
        
        ft_ack.change(_ack_ft_cfg, [ft_ack], [ft_start_row])
                    
        def _begin_finetuning():
            configs = utils.retrieve_configs()
            rc = subprocess.call("/bin/bash /project/code/scripts/run_peft.sh" + 
                                 " LOG_EVERY_N_STEPS=" + str(configs["log_every_n_steps"]) + 
                                 " PRECISION=" + str(configs["precision"]) + 
                                 " DEVICES=" + str(configs["devices"]) + 
                                 " NUM_NODES=" + str(configs["num_nodes"]) + 
                                 " VAL_CHECK_INTERVAL=" + str(configs["val_check_interval"]) + 
                                 " MAX_STEPS=" + str(configs["max_steps"]) + 
                                 " RESTORE_FROM_PATH=" + str(configs["restore_from_path"]) +
                                 " PEFT_SCHEME=" + str(configs["peft_scheme"]) + 
                                 " MICRO_BATCH_SIZE=" + str(configs["micro_batch_size"]) + 
                                 " GLOBAL_BATCH_SIZE=" + str(configs["global_batch_size"]) + 
                                 " TENSOR_MODEL_PARALLEL_SIZE=" + str(configs["tensor_model_parallel_size"]) + 
                                 " PIPELINE_MODEL_PARALLEL_SIZE=" + str(configs["pipeline_model_parallel_size"]) + 
                                 " MEGATRON_AMP_02=" + str(configs["megatron_amp_O2"]) + 
                                 " ACTIVATIONS_CHECKPOINT_GRANULARITY=" + str(configs["activations_checkpoint_granularity"]) + 
                                 " ACTIVATIONS_CHECKPOINT_NUM_LAYERS=" + str(configs["activations_checkpoint_num_layers"]) + 
                                 " ACTIVATIONS_CHECKPOINT_METHOD=" + str(configs["activations_checkpoint_method"]) + 
                                 " OPTIM_NAME=" + str(configs["optim_name"]) + 
                                 " LR=" + str(configs["lr"]) + 
                                 " ANSWER_ONLY_LOSS=" + str(configs["answer_only_loss"]) + 
                                 " TRAIN_FILE_NAMES=" + str(configs["train_file_names"]) + 
                                 " TRAIN_CONCAT_SAMPLING_PROBABILITIES=" + str(configs["train_concat_sampling_probabilities"]) + 
                                 " TRAIN_MAX_SEQ_LENGTH=" + str(configs["train_max_seq_length"]) + 
                                 " TRAIN_MICRO_BATCH_SIZE=" + str(configs["train_micro_batch_size"]) + 
                                 " TRAIN_GLOBAL_BATCH_SIZE=" + str(configs["train_global_batch_size"]) + 
                                 " TRAIN_NUM_WORKERS=" + str(configs["train_num_workers"]) + 
                                 " VAL_FILE_NAMES=" + str(configs["val_file_names"]) + 
                                 " VAL_MAX_SEQ_LENGTH=" + str(configs["val_max_seq_length"]) + 
                                 " VAL_MICRO_BATCH_SIZE=" + str(configs["val_micro_batch_size"]) + 
                                 " VAL_GLOBAL_BATCH_SIZE=" + str(configs["val_global_batch_size"]) + 
                                 " VAL_NUM_WORKERS=" + str(configs["val_num_workers"]) + 
                                 " VAL_NAME=" + str(configs["val_name"]) + 
                                 " TEST_FILE_NAMES=" + str(configs["test_file_names"]) + 
                                 " TEST_NUM_WORKERS=" + str(configs["test_num_workers"]) + 
                                 " TEST_NAME=" + str(configs["test_name"]) + 
                                 " SAVE_NEMO_ON_VALIDATION_END=" + str(configs["save_nemo_on_validation_end"]) + 
                                 " CREATE_WANDB_LOGGER=" + str(configs["create_wandb_logger"]) + 
                                 " CHECKPOINT_CALLBACK_PARAMS_MODE=" + str(configs["checkpoint_callback_params_mode"]) + 
                                 " EXPLICIT_LOG_DIR=" + str(configs["explicit_log_dir"]) + 
                                 " RESUME_IF_EXISTS=" + str(configs["resume_if_exists"]) + 
                                 " RESUME_IGNORE_NO_CHECKPOINT=" + str(configs["resume_ignore_no_checkpoint"]) + 
                                 " CREATE_CHECKPOINT_CALLBACK=" + str(configs["create_checkpoint_callback"]) + 
                                 " MONITOR=" + str(configs["monitor"]) + 
                                 " SAVE_BEST_MODEL=" + str(configs["save_best_model"]) + 
                                 " SAVE_NEMO_ON_TRAIN_END=" + str(configs["save_nemo_on_train_end"]) + 
                                 " >> " + "/project/models/output.log", shell=True)
            if rc == 0:
                return {
                    ft_start: gr.update(value="Finetuning Complete", variant="primary", interactive=False), 
                }
            else:
                return {
                    ft_start: gr.update(variant="secondary", interactive=True), 
                }

        ft_start.click(_begin_finetuning, [], [ft_start])
        
        @undo_btn.click(outputs=editor)
        def read_chain_config() -> str:
            """Read the chain config file."""
            with open(config.chain_config_file, "r", encoding="UTF-8") as cf:
                return cf.read()

        # pylint: disable-next=no-member # false positive
        @reset_btn.click(outputs=editor)
        def reset_demo() -> str:
            """Reset the configuration to the starting config."""
            return _STARTING_CONFIG

        # pylint: disable-next=no-member # false positive
        @save_btn.click(inputs=editor)
        def save_chain_config(config_txt: str) -> None:
            """Save the user's config file."""
            # validate yaml
            try:
                config_data = yaml.safe_load(config_txt)
            except Exception as err:
                raise SyntaxError(f"Error validating YAML syntax:\n{err}") from err

            # save configuration
            with open(config.chain_config_file, "w", encoding="UTF-8") as cf:
                cf.write(config_txt)

        # pylint: disable-next=no-member # false positive
        editor.input(None, js=_CONFIG_CHANGES_JS)
        # pylint: disable-next=no-member # false positive
        save_btn.click(None, js=_SAVE_CHANGES_JS)
        # pylint: disable-next=no-member # false positive
        undo_btn.click(None, js=_SAVE_CHANGES_JS)
    
    def _verify_settings(selected_model: str, selected_train_ds: str, selected_val_ds: str):
        if len(selected_model) != 0: # model has been loaded
            if (len(selected_train_ds) != 0 and len(selected_val_ds) != 0): # dataset has been loaded
                return {
                    ft_not_ready: gr.update(visible=False),
                    ft_ready: gr.update(visible=True),
                    err_msg: gr.update(value="You are not ready to start finetuning: Load in the models and select a dataset."),
                }
            else:
                return {
                    ft_not_ready: gr.update(visible=True),
                    ft_ready: gr.update(visible=False),
                    err_msg: gr.update(value="You are not ready to start finetuning: Finish configuring your dataset."),
                }
        else: 
            return {
                ft_not_ready: gr.update(visible=True),
                ft_ready: gr.update(visible=False),
                err_msg: gr.update(value="You are not ready to start finetuning: You are missing a model."),
            }
        return {
            ft_not_ready: gr.update(visible=True),
            ft_ready: gr.update(visible=False),
            err_msg: gr.update(value="You are not ready to start finetuning: Load in the models and select a dataset."),
        }
    
    ft_tab.select(_verify_settings, [selected_model, selected_train_ds, selected_val_ds], [ft_not_ready, ft_ready, err_msg])
    
    with gr.Tab("Evaluation", elem_classes=["invert-bg"]) as eval_tab:
        with gr.Row(elem_id="config-row", visible=True) as eval_not_ready:
            eval_err_msg = gr.Markdown("You are not ready to start evaluation: Unable to detect the finetuned model.")
        with gr.Row(elem_id="config-row", visible=False) as eval_ready:
            with gr.Column():
                with gr.Row(elem_id="config-row"):
                    with gr.Column(scale=1):
                        gr.Markdown(markdown.CONFIG_EVAL_MARKDOWN)
                    with gr.Column(scale=1):
                        with gr.Row(elem_id="config-row"):
                            gr.Markdown(markdown.EVAL_MARKDOWN)
                        with gr.Row(elem_id="config-row"):
                            eval_ack = gr.Checkbox(label="I have finished the below configuration file.")
                        with gr.Row(elem_id="config-row", visible=False) as eval_start_row:
                            eval_start = gr.Button("Start Evaluation", size="sm", scale=1)
                        gr.Markdown("<br />")
                        with gr.Row(elem_id="config-row") as eval_file:
                            eval_file = gr.File(visible=False, interactive=False)
                with gr.Row(elem_id="config-row"):
                    with gr.Group(elem_id="config-wrapper"):
                        with gr.Row(elem_id="config-toolbar", elem_classes=["toolbar"]):
                            eval_file_title = gr.Markdown(_PSEUDO_FILE_EVAL_NAME, elem_id="editor-title")
                            eval_save_btn = gr.Button("", icon=_SAVE_IMG, elem_classes=["toolbar"])
                            eval_undo_btn = gr.Button("", icon=_UNDO_IMG, elem_classes=["toolbar"])
                            eval_reset_btn = gr.Button("", icon=_HISTORY_IMG, elem_classes=["toolbar"])
                        with gr.Row(elem_id="config-row-box"):
                            eval_editor = gr.Code(
                                _STARTING_EVAL_CONFIG,
                                elem_id="config-editor",
                                interactive=True,
                                language="yaml",
                                show_label=False,
                                container=False,
                            )
                            
        def _ack_eval_cfg(eval_ack):
            return {
                eval_start_row: gr.update(visible=eval_ack)
            }
        
        eval_ack.change(_ack_eval_cfg, [eval_ack], [eval_start_row])
                            
        @eval_undo_btn.click(outputs=eval_editor)
        def read_chain_config_eval() -> str:
            """Read the chain config file."""
            with open("/project/code/configs/config-eval.yaml", "r", encoding="UTF-8") as cf:
                return cf.read()

        # pylint: disable-next=no-member # false positive
        @eval_reset_btn.click(outputs=eval_editor)
        def reset_demo_eval() -> str:
            """Reset the configuration to the starting config."""
            return _STARTING_EVAL_CONFIG

        # pylint: disable-next=no-member # false positive
        @eval_save_btn.click(inputs=eval_editor)
        def save_chain_config_eval(config_txt: str) -> None:
            """Save the user's config file."""
            # validate yaml
            try:
                config_data = yaml.safe_load(config_txt)
            except Exception as err:
                raise SyntaxError(f"Error validating YAML syntax:\n{err}") from err

            # save configuration
            with open("/project/code/configs/config-eval.yaml", "w", encoding="UTF-8") as cf:
                cf.write(config_txt)

        # pylint: disable-next=no-member # false positive
        eval_editor.input(None, js=_CONFIG_CHANGES_JS)
        # pylint: disable-next=no-member # false positive
        eval_save_btn.click(None, js=_SAVE_CHANGES_JS)
        # pylint: disable-next=no-member # false positive
        eval_undo_btn.click(None, js=_SAVE_CHANGES_JS)
        
        def _begin_evaluation():
            configs = utils.retrieve_eval_configs()
            rc = subprocess.call("/bin/bash /project/code/scripts/run_eval.sh" + 
                                 " DEVICES=" + str(configs["devices"]) + 
                                 " RESTORE_FROM_PATH=" + str(configs["restore_from_path"]) + 
                                 " PEFT_RESTORE_FROM_PATH=" + str(configs["peft_restore_from_path"]) + 
                                 " TENSOR_MODEL_PARALLEL_SIZE=" + str(configs["tensor_model_parallel_size"]) + 
                                 " PIPELINE_MODEL_PARALLEL_SIZE=" + str(configs["pipeline_model_parallel_size"]) + 
                                 " MEGATRON_AMP_02=" + str(configs["megatron_amp_O2"]) + 
                                 " ANSWER_ONLY_LOSS=" + str(configs["answer_only_loss"]) + 
                                 " FILE_NAMES=" + str(configs["file_names"]) + 
                                 " NAMES=" + str(configs["names"]) + 
                                 " MICRO_BATCH_SIZE=" + str(configs["micro_batch_size"]) + 
                                 " GLOBAL_BATCH_SIZE=" + str(configs["global_batch_size"]) + 
                                 " TOKENS_TO_GENERATE=" + str(configs["tokens_to_generate"]) + 
                                 " OUTPUT_FILE_PATHS_PREFIX=" + str(configs["output_file_path_prefix"]) + 
                                 " WRITE_PREDICTIONS_TO_FILE=" + str(configs["write_predictions_to_file"]) + 
                                 " GREEDY=" + str(configs["greedy"]) + 
                                 " >> " + "/project/models/output.log", shell=True)
            with open("/project/code/configs/config-eval.yaml", "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            prefix = config["model"]["data"]["test_ds"]["output_file_path_prefix"]
            names = config["model"]["data"]["test_ds"]["names"].strip('][').split(',')
            files = [(prefix + '_test_' + name.strip() + "_inputs_preds_labels.jsonl") for name in names]
            if rc == 0:
                return {
                    eval_start: gr.update(value="Evaluation Complete", variant="primary", interactive=False), 
                    eval_file: gr.update(value=files, visible=True)
                }
            else:
                return {
                    eval_start: gr.update(variant="secondary", interactive=True), 
                }

        eval_start.click(_begin_evaluation, [], [eval_start, eval_file])
                
        def _verify_eval_settings(selected_model: str, selected_train_ds: str, selected_val_ds: str):
            for fname in os.listdir("/project/models/results/checkpoints"):
                if fname.endswith('.nemo'):
                    if len(selected_model) != 0: # model has been loaded
                        if (len(selected_train_ds) != 0 and len(selected_val_ds) != 0): # dataset has been loaded
                            return {
                                eval_not_ready: gr.update(visible=False),
                                eval_ready: gr.update(visible=True),
                                eval_err_msg: gr.update(value="You are not ready to start evaluation: Load in the models and select a dataset."),
                            }
                        else:
                            return {
                                eval_not_ready: gr.update(visible=True),
                                eval_ready: gr.update(visible=False),
                                eval_err_msg: gr.update(value="You are not ready to start evaluation: Finish configuring your dataset."),
                            }
                    else: 
                        return {
                            eval_not_ready: gr.update(visible=True),
                            eval_ready: gr.update(visible=False),
                            eval_err_msg: gr.update(value="You are not ready to start evaluation: You are missing a model."),
                        }
            else:
                return {
                    eval_not_ready: gr.update(visible=True),
                    eval_ready: gr.update(visible=False),
                    eval_err_msg: gr.update(value="You are not ready to start evaluation: Unable to detect the finetuned model."),
                }

        eval_tab.select(_verify_eval_settings, [selected_model, selected_train_ds, selected_val_ds], [eval_not_ready, eval_ready, eval_err_msg])