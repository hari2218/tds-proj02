from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, Optional
import json
import shutil
import zipfile
import pandas as pd
import os
import subprocess
import ast
import importlib
import importlib.metadata as metadata
import ast
import subprocess
import pkg_resources
import logging
import httpx
import re

app = FastAPI()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

ROOT_DIR: str = "./tds2025_temp"
DEV_EMAIL: str = "hariharan.chandran@straive.com"

# AI Proxy
# # AI_URL: str = "https://api.openai.com/v1"
AI_URL: str = "https://aiproxy.sanand.workers.dev/openai/v1"
AIPROXY_TOKEN: str = ""  # os.environ.get("AIPROXY_TOKEN")
AI_MODEL: str = "gpt-4o-mini"
AI_EMBEDDINGS_MODEL: str = "text-embedding-3-small"

# for debugging use LLM token
if not AIPROXY_TOKEN:
    AI_URL = "https://llmfoundry.straive.com/openai/v1"
    AIPROXY_TOKEN = os.environ.get("LLM_TOKEN")

if not AIPROXY_TOKEN:
    raise KeyError("AIPROXY_TOKEN environment variables is missing")

APP_ID = "tds-proj02"
ssl_verify = False

DATA_DIR = f"{ROOT_DIR}/data"
SCRIPT_DIR = f"{ROOT_DIR}/scripts"

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)


def get_tool_calls(tool: Dict[str, Any]):
    if tool and "tool_calls" in tool:
        for tool_call in tool["tool_calls"]:
            function_name = tool_call["function"].get("name")
            function_args = tool_call["function"].get("arguments")

            # Ensure the function name is valid and callable
            if function_name in globals() and callable(globals()[function_name]):
                function_chosen = globals()[function_name]
                function_args = parse_function_args(function_args)

                return (
                    function_chosen,
                    function_args,
                )

    raise NotImplementedError("Unknown task")


def execute_tool_calls(tool: Dict[str, Any]):
    function_chosen, function_args = get_tool_calls(tool)

    if isinstance(function_args, dict):
        return function_chosen(**function_args)

    else:
        return function_chosen()


def parse_function_args(function_args: Optional[Any]):
    if function_args is not None:
        if isinstance(function_args, str):
            function_args = json.loads(function_args)

        elif not isinstance(function_args, dict):
            function_args = {"args": function_args}

    else:
        function_args = {}

    return function_args


# Task implementations
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_script",
            "description": "Any task",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": ["string", "null"],
                        "description": "The file for the task to be performed. If unavailable, set to null.",
                        "nullable": True,
                    }
                },
                "required": ["file"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


def get_task_tool(task: str, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}:{APP_ID}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": task}],
            "tools": tools,
            "tool_choice": "auto",
        },
        verify=ssl_verify,
    )

    # response.raise_for_status()

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["choices"][0]["message"]


def get_chat_completions(messages: list[Dict[str, Any]]):
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}:{APP_ID}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "messages": messages,
        },
        verify=ssl_verify,
    )

    # response.raise_for_status()

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["choices"][0]["message"]


def get_embeddings(text: str):
    response = httpx.post(
        f"{AI_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}:{APP_ID}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_EMBEDDINGS_MODEL,
            "input": text,
        },
        verify=ssl_verify,
    )

    # response.raise_for_status()

    json_response = response.json()

    if "error" in json_response:
        raise HTTPException(status_code=500, detail=json_response["error"]["message"])

    return json_response["data"][0]["embedding"]


# P2
def extract_imports(script_path):
    """Extracts module names from a given Python script."""
    with open(script_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=script_path)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    return imports


def check_modules(modules):
    """Checks if modules are installed and suggests installation or upgrade."""
    missing_modules = []
    outdated_modules = []

    for module in modules:
        try:
            pkg = importlib.import_module(module)
            dist = metadata.distribution(module)
            outdated_modules.append(f"{module}=={dist.version}")

        except ImportError:
            missing_modules.append(module)

        except pkg_resources.DistributionNotFound:
            missing_modules.append(module)

    return missing_modules, outdated_modules


@app.post("/api")
async def process_file(
    question: str = Form(...), file: Optional[UploadFile] = File(None)
):
    try:
        downloaded_file = None

        if not question:
            raise HTTPException(status_code=400, detail="Task description is required")

        # Save uploaded file
        if file and file.filename:
            downloaded_file = os.path.join(DATA_DIR, file.filename)

            with open(downloaded_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        tool = get_task_tool(question, tools)
        function_chosen, function_args = get_tool_calls(tool)

        if not function_args:
            function_args = {}

        # if downloaded_file:
        #     function_args["downloaded_file"] = downloaded_file

        return function_chosen(question, downloaded_file, **function_args)

        # script_content = ""

        # script_path = os.path.join(SCRIPT_DIR, "script.py")
        # with open(script_path, "w") as script_file:
        #     script_file.write(script_content)

        # modules = extract_imports(script_path)
        # missing, outdated = check_modules(modules)

        # for module in missing:
        #     subprocess.run(
        #         ["uv", "pip", "install", module], capture_output=True, text=True
        #     )

        # # Execute script
        # result = subprocess.run(["python", script_path], capture_output=True, text=True)
        # output = result.stdout.strip()

        # if not output:
        #     return {
        #         "error": "I am having difficulty obtaining results from the script."
        #     }

        # return {"answer": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_script(task: str, downloaded_file: str, **args):
    if downloaded_file and "file" in args and args["file"]:
        arg_file = args["file"]
        task = task.replace(arg_file, downloaded_file)

        arg_file = re.sub(
            r"download_file\(['\"](.+?)['\"]\)", r"\1", arg_file, flags=re.DOTALL
        )
        task = task.replace(arg_file, downloaded_file)

    response = get_chat_completions(
        [
            {
                "role": "system",
                "content": "Write a python script for the following task, script should be well returned without ant structural or logical error. Only result should be printed in the output. And any error should be handled accordingly. Return only the python script.",
            },
            {"role": "user", "content": task},
        ]
    )

    script_content = response["content"].strip()

    # get only the python code block, using re
    script_content = re.sub(
        r"```python[\r\n]*(.+?)[\r\n]*```", r"\1", script_content, flags=re.DOTALL
    )

    script_path = os.path.join(SCRIPT_DIR, "script.py")
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    modules = extract_imports(script_path)
    missing, outdated = check_modules(modules)

    for module in missing:
        subprocess.run(["uv", "pip", "install", module], capture_output=True, text=True)

    # Execute script
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    output = result.stdout.strip()

    return {
        "answer": output,
    }
