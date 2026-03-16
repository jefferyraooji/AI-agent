import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "workspace")).resolve()

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are a file assistant that converts a user's request into ONE JSON action.

Return ONLY valid JSON.
No markdown.
No explanation.

Allowed actions:
- list_files
- create_file
- add_to_file
- edit_file
- delete_file
- rename_file
- summarize_file
- read_file

Rules:
- Use only relative paths.
- Never use absolute paths.
- Never use '..'.
- If a field is unused, return an empty string.
- For create_file, put the initial content into "content".
- For add_to_file, put appended text into "content".
- For edit_file, put the edit request into "instruction".

Required JSON shape:
{
  "action": "string",
  "file_path": "string",
  "new_file_path": "string",
  "content": "string",
  "instruction": "string"
}
"""

ACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": [
                "list_files",
                "create_file",
                "add_to_file",
                "edit_file",
                "delete_file",
                "rename_file",
                "summarize_file",
                "read_file",
            ],
        },
        "file_path": {"type": "STRING"},
        "new_file_path": {"type": "STRING"},
        "content": {"type": "STRING"},
        "instruction": {"type": "STRING"},
    },
    "required": ["action", "file_path", "new_file_path", "content", "instruction"],
}

# This function protects the filesystem.
def safe_path(relative_path: str) -> Path:
    if not relative_path:
        return WORKSPACE_DIR

    if ".." in relative_path:
        raise ValueError("Path traversal is not allowed.")

    full_path = (WORKSPACE_DIR / relative_path).resolve()

    if full_path != WORKSPACE_DIR and WORKSPACE_DIR not in full_path.parents:
        raise ValueError("Access outside workspace is not allowed.")

    return full_path

# This is the low-level function that sends a request to Gemini.
def post_gemini(payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# Gemini does not return plain text directly. It returns a nested response object.
# This function extracts the text safely from that structure.
def extract_text_from_gemini_response(data: Dict[str, Any], allow_empty: bool = False) -> str:
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            if allow_empty:
                return ""
            raise ValueError("No candidates returned.")

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        if not parts:
            if allow_empty:
                return ""
            raise ValueError("No text parts returned.")

        texts = []
        for part in parts:
            text = part.get("text")
            if text is not None:
                texts.append(text)

        if not texts:
            if allow_empty:
                return ""
            raise ValueError("No text field found in response parts.")

        return "".join(texts)
    except Exception as e:
        raise ValueError(f"Unexpected Gemini response:\n{json.dumps(data, indent=2)}") from e

# Convert the user’s natural-language command into structured action JSON.
# system prompt + user prompt + output format
# This is the planner part of the agent.
def gemini_generate_json(user_command: str) -> Dict[str, Any]:
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_command}]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": ACTION_SCHEMA,
        }
    }

    data = post_gemini(payload, timeout=60)
    text = extract_text_from_gemini_response(data, allow_empty=False)

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{text}") from e

    for key in ["action", "file_path", "new_file_path", "content", "instruction"]:
        if key not in result:
            raise ValueError(f"Missing key in model output: {key}")

    return result

# This function is used when the user wants to edit a file.
def gemini_rewrite_file(old_content: str, instruction: str) -> str:
    payload = {
        "system_instruction": {
            "parts": [{
                "text": (
                    "Return only the full revised file content. "
                    "No markdown fences. No explanation. "
                    "If the instruction asks to remove all content, return an empty response."
                )
            }]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": f"Instruction:\n{instruction}\n\nCurrent file content:\n{old_content}"
                }]
            }
        ],
        "generationConfig": {
            "temperature": 0.2
        }
    }

    data = post_gemini(payload, timeout=120)
    return extract_text_from_gemini_response(data, allow_empty=True)

# This function sends file content to Gemini and asks for a brief summary.
def gemini_summarize(content: str) -> str:
    payload = {
        "system_instruction": {
            "parts": [{"text": "Summarize this file clearly and briefly."}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": content}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2
        }
    }

    data = post_gemini(payload, timeout=120)
    return extract_text_from_gemini_response(data, allow_empty=False).strip()

# This lists all files inside the workspace recursively.
def list_files() -> str:
    items = []
    for path in WORKSPACE_DIR.rglob("*"):
        if path.is_file():
            items.append(str(path.relative_to(WORKSPACE_DIR)))
    if not items:
        return "Workspace is empty."
    return "Files:\n" + "\n".join(sorted(items))

# This creates a file and writes content to it.
def create_file(file_path: str, content: str) -> str:
    path = safe_path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Created file: {file_path}"

# This appends content to an existing file.
def add_to_file(file_path: str, content: str) -> str:
    path = safe_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size > 0 and content:
            f.write("\n")
        f.write(content)
    return f"Added content to: {file_path}"

# This reads the full contents of a file and returns it.
def read_file(file_path: str) -> str:
    path = safe_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding="utf-8")

# This deletes a file.
def delete_file(file_path: str) -> str:
    path = safe_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    path.unlink()
    return f"Deleted file: {file_path}"

# This renames or moves a file.
def rename_file(file_path: str, new_file_path: str) -> str:
    old_path = safe_path(file_path)
    new_path = safe_path(new_file_path)
    if not old_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.rename(new_path)
    return f"Renamed {file_path} -> {new_file_path}"

# This is the most interesting tool because it combines local code and Gemini.
def edit_file(file_path: str, instruction: str) -> str:
    path = safe_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    normalized = instruction.strip().lower()
    clear_commands = {
        "remove all content",
        "clear file",
        "empty file",
        "delete all content",
        "remove everything",
        "clear all content",
    }

    if normalized in clear_commands:
        path.write_text("", encoding="utf-8")
        return f"Cleared file: {file_path}"

    old_content = path.read_text(encoding="utf-8")
    new_content = gemini_rewrite_file(old_content, instruction)
    path.write_text(new_content, encoding="utf-8")
    return f"Edited file: {file_path}"

# This reads a file and then asks Gemini to summarize it.
def summarize_file(file_path: str) -> str:
    path = safe_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8")
    return gemini_summarize(content)

# This is the dispatcher.
# It takes the structured JSON from Gemini and routes it to the correct local function.
def execute_action(action_data: Dict[str, Any]) -> str:
    action = action_data["action"]
    file_path = action_data["file_path"]
    new_file_path = action_data["new_file_path"]
    content = action_data["content"]
    instruction = action_data["instruction"]

    if action == "list_files":
        return list_files()
    if action == "create_file":
        return create_file(file_path, content)
    if action == "add_to_file":
        return add_to_file(file_path, content)
    if action == "edit_file":
        return edit_file(file_path, instruction)
    if action == "delete_file":
        return delete_file(file_path)
    if action == "rename_file":
        return rename_file(file_path, new_file_path)
    if action == "summarize_file":
        return summarize_file(file_path)
    if action == "read_file":
        return read_file(file_path)

    raise ValueError(f"Unknown action: {action}")

# This is the main interactive loop of the whole program.
def main():
    print("Simple Gemini File Agent (REST version)")
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"Model: {MODEL}")
    print("Type 'exit' to quit.\n")

    while True:
        user_command = input("You> ").strip()
        if user_command.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            action_data = gemini_generate_json(user_command)

            print("\n[Planned Action]")
            print(json.dumps(action_data, indent=2, ensure_ascii=False))

            result = execute_action(action_data)

            print("\n[Result]")
            print(result)
            print()

        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    main()