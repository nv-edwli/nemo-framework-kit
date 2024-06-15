from huggingface_hub import snapshot_download
import os
import sys

snapshot_download(
    repo_id="meta-llama/Llama-2-" + sys.argv[1] + "-hf",
    local_dir="/project/models/llama2-" + sys.argv[1] + "-hf",
    local_dir_use_symlinks=False,
    token=os.environ['HF_API_KEY']
)