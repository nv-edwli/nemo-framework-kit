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

"""A page to display the output console to the user."""

from ...common import THEME
from ..finetune import utils
from .logger import Logger

from pathlib import Path
import gradio as gr
import sys

# load custom style and scripts
_CSS_FILE = Path(__file__).parent.joinpath("style.css")
_CSS = open(_CSS_FILE, "r", encoding="UTF-8").read()

sys.stdout = Logger("/project/models/output.log")

with gr.Blocks(theme=THEME, css=_CSS) as page:
    logs = gr.Textbox(label="Output Console", lines=40, max_lines=40, interactive=False)
    
    page.load(utils.read_logs, None, logs, every=1)