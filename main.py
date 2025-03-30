from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, Request
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
import stat
import time

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
AIPROXY_TOKEN: str = os.environ.get("AIPROXY_TOKEN")
AI_MODEL: str = "gpt-4o-mini"
AI_EMBEDDINGS_MODEL: str = "text-embedding-3-small"

# for debugging use LLM token
# if not AIPROXY_TOKEN:
#     AI_URL = "https://llmfoundry.straive.com/openai/v1"
#     AIPROXY_TOKEN = os.environ.get("LLM_TOKEN")

if not AIPROXY_TOKEN:
    raise KeyError("AIPROXY_TOKEN environment variables is missing")

APP_ID = "tds-proj02"
ssl_verify = False

os.makedirs(ROOT_DIR, exist_ok=True)


def get_tool_calls(tool: Dict[str, Any]):
    if tool:
        if "tool_calls" in tool:
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

        if "content" in tool and tool["content"]:
            raise ValueError(tool["content"])

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
            "name": "google_sheet_formula",
            "description": "Task to evaluate Google Sheet formula",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "Google Sheet formula",
                    }
                },
                "required": ["formula"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "excel_formula",
            "description": "Task to evaluate Excel formula",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "Excel formula",
                    }
                },
                "required": ["formula"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_script",
            "description": "Task on a file or a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task to be performed",
                        "nullable": True,
                    },
                    "file": {
                        "type": ["string", "null"],
                        "description": "The file to be downloaded. If unavailable, set to null.",
                        "nullable": True,
                    },
                    "url": {
                        "type": ["string", "null"],
                        "description": "The URL to be accessed. If unavailable, set to null.",
                        "nullable": True,
                    },
                },
                "required": ["task", "file", "url"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_script",
            "description": "Task to perfome on an attached file",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task to be performed",
                        "nullable": True,
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
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
    question: str = Form(...),
    file: Optional[UploadFile] = File(None),
    request: Request = None,
):
    uid = str(int(time.time()))

    data_dir = os.path.join(ROOT_DIR, f"data_{uid}")

    script_dir = os.path.join(ROOT_DIR, "script")
    os.makedirs(script_dir, exist_ok=True)

    script_file = os.path.join(ROOT_DIR, "script", f"task_{uid}.py")

    try:
        downloaded_file = None

        if not question:
            raise HTTPException(status_code=400, detail="Task description is required")

        # Save uploaded file
        if file and file.filename:
            os.makedirs(data_dir, exist_ok=True)

            downloaded_file = os.path.join(data_dir, file.filename)

            with open(downloaded_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        tool = get_task_tool(question, tools)
        function_chosen, function_args = get_tool_calls(tool)

        if not function_args:
            function_args = {}

        if function_chosen.__name__ in {
            "create_script",
            "create_script1",
            "create_script2",
        }:
            if "task" in function_args:
                # question = function_args["task"]
                del function_args["task"]

            return create_script(
                question, script_file, downloaded_file, **function_args
            )

        else:
            return function_chosen(**function_args)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def create_script(task: str, script_path: str, downloaded_file: str, **args):
    if downloaded_file:
        downloaded_file = os.path.abspath(downloaded_file)

        if "file" in args and args["file"]:
            arg_file = args["file"]
            task = task.replace(arg_file, downloaded_file)

            arg_file = re.sub(
                r"download_file\(['\"](.+?)['\"]\)", r"\1", arg_file, flags=re.DOTALL
            )
            task = task.replace(arg_file, downloaded_file)
            task = re.sub(r"\b(download)\b", " ", task, flags=re.IGNORECASE)

        else:
            file_name = os.path.basename(downloaded_file)
            task = task.replace(file_name, downloaded_file)

    file_prompt = f"\n\nFile: {downloaded_file}" if downloaded_file else ""

    response = get_chat_completions(
        [
            {
                "role": "system",
                "content": """
Develop a Python script to perform the specified task, adhering to the following requirements:  

- Use the latest stable versions of any required modules or libraries, ensuring compatibility with the most recent Python release.  
- The script should be free of structural, logical, or syntax errors.  
- The output must include only the result or any error message, with no additional explanations, debug information, or extraneous values.  
- If the result of the task is a file, print only the file path.  
- Handle exceptions robustly and take appropriate actions in each case to ensure graceful error management without unexpected terminations.  
- Adhere to Python's best practices (e.g., PEP 8 style guide) for clarity and maintainability.  
- Ensure all instructions in this prompt are implemented in the script comprehensively and without omissions.  

Return only the Python script as the response, without any additional explanations or text.
""".strip(),
            },
            {"role": "user", "content": task + file_prompt},
        ]
    )

    script_content = response["content"].strip()

    # get only the python code block, using re
    script_content = re.sub(
        r"```python[\r\n]*(.+?)[\r\n]*```", r"\1", script_content, flags=re.DOTALL
    )

    # with open(script_path + ".txt", "w") as script_file:
    #     script_file.write(task)

    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    modules = extract_imports(script_path)
    missing, outdated = check_modules(modules)

    for module in missing:
        subprocess.run(["uv", "pip", "install", module], capture_output=True, text=True)

    # Execute script
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    if result.stderr:
        raise HTTPException(status_code=500, detail=result.stderr)

    out_str = result.stdout.strip()

    # if out_str is absulete path make it relative path
    if os.path.isabs(out_str):
        out_str = os.path.relpath(out_str, ROOT_DIR)

    out_str = out_str.replace(ROOT_DIR, ".")

    return {
        "answer": result.stdout.strip(),
    }


def google_sheet_formula(formula: str):
    response = get_chat_completions(
        [
            {
                "role": "system",
                "content": "What would be the output for the following formula in Google Sheet? Return only the final value.",
            },
            {"role": "user", "content": formula},
        ]
    )

    return {"answer": response["content"].strip()}


def excel_formula(formula: str):
    response = get_chat_completions(
        [
            {
                "role": "system",
                "content": "What would be the output for the following formula in Excel? Return only the final value.",
            },
            {"role": "user", "content": formula},
        ]
    )

    return {"answer": response["content"].strip()}
