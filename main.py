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

DATA_DIR: str = "/data"
DEV_EMAIL: str = "hariharan.chandran@straive.com"

# AI Proxy
# # AI_URL: str = "https://api.openai.com/v1"
AI_URL: str = "https://aiproxy.sanand.workers.dev/openai/v1"
AIPROXY_TOKEN: str = os.environ.get("AIPROXY_TOKEN")
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

UPLOAD_DIR = f"{DATA_DIR}/uploads"
EXTRACT_DIR = f"{DATA_DIR}/extracted"
SCRIPT_DIR = f"{DATA_DIR}/scripts"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)


def execute_tool_calls(tool: Dict[str, Any]):
    if tool and "tool_calls" in tool:
        for tool_call in tool["tool_calls"]:
            function_name = tool_call["function"].get("name")
            function_args = tool_call["function"].get("arguments")

            # Ensure the function name is valid and callable
            if function_name in globals() and callable(globals()[function_name]):
                function_chosen = globals()[function_name]
                function_args = parse_function_args(function_args)

                if isinstance(function_args, dict):
                    return function_chosen(**function_args)

                else:
                    return function_chosen()

    raise NotImplementedError("Unknown task")


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
tools = {
    "type": "function",
    "function": {
        "name": "get_script",
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


def get_task_tool(task: str, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    response = httpx.post(
        f"{AI_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
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
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
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
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
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


@app.post("/api/")
async def process_file(question: str, file: UploadFile = File(...)):
    try:
        if not question:
            raise ValueError("Task description is required")

        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # # Extract ZIP
        # with zipfile.ZipFile(file_path, "r") as zip_ref:
        #     zip_ref.extractall(EXTRACT_DIR)

        # # Find CSV File
        # csv_file = next((f for f in os.listdir(EXTRACT_DIR) if f.endswith(".csv")), None)
        # if not csv_file:
        #     return {"error": "No CSV file found in ZIP"}

        # # Read CSV
        # csv_path = os.path.join(EXTRACT_DIR, csv_file)
        # df = pd.read_csv(csv_path)

        # if "answer" not in df.columns:
        #     return {"error": "No 'answer' column found in CSV"}

        # # Get answer value (assuming first row)
        # answer = df["answer"].iloc[0]

        # return {"answer": str(answer)}

        script_content = """
import requests
import zipfile
import os

# URL of the ZIP file
url = "https://example.com/abcd.zip"  # Replace with actual URL

# Local filename to save the ZIP file
zip_file_path = "abcd.zip"

# Download the ZIP file
print("Downloading...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print("Download complete.")
else:
    print("Failed to download file.")
    exit()

# Unzip the file
extract_folder = "abcd_extracted"
os.makedirs(extract_folder, exist_ok=True)

print("Extracting...")
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(extract_folder)
print("Extraction complete.")

# Optionally, delete the ZIP file after extraction
os.remove(zip_file_path)
print("Cleanup complete.")

"""

        script_path = os.path.join(SCRIPT_DIR, "script.py")
        with open(script_path, "w") as script_file:
            script_file.write(script_content)

        modules = extract_imports(script_path)
        missing, outdated = check_modules(modules)

        for module in missing:
            subprocess.run(
                ["uv", "pip", "install", module], capture_output=True, text=True
            )

        # Execute script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        output = result.stdout.strip()

        if not output:
            return {
                "error": "I am having difficulty obtaining results from the script."
            }

        return {"answer": output}

    except Exception as e:
        return {"error": str(e)}
