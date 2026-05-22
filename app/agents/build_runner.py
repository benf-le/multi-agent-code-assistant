import json
import os
import subprocess
import threading
import queue
import time
import re
from pathlib import Path
from typing import Dict, Any, Callable, Optional

def _parse_project_context(project_context_str: str | None) -> dict:
    if not project_context_str:
        return {}
    try:
        json_str = project_context_str
        if "## project_context.json" in json_str:
            parts = json_str.split("## project_context.json")
            if len(parts) > 1:
                json_str = parts[1].strip()
        
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
            
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
            return {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def _detect_language(candidate_path: Path, ctx: dict) -> str:
    language = ctx.get("stack", {}).get("language", "unknown")
    if language != "unknown":
        return language
    
    if (candidate_path / "package.json").exists():
        return "Node.js"
    if (candidate_path / "requirements.txt").exists() or (candidate_path / "pytest.ini").exists():
        return "Python"
    if (candidate_path / "go.mod").exists():
        return "Go"
    if list(candidate_path.glob("*.csproj")) or list(candidate_path.glob("*.sln")):
        return ".NET"
    if (candidate_path / "pom.xml").exists():
        return "Java"
    if (candidate_path / "Cargo.toml").exists():
        return "Rust"
    return "unknown"

def _detect_install_cmd(candidate_path: Path, ctx: dict) -> str | None:
    install_cmd = ctx.get("run_commands", {}).get("install")
    if install_cmd:
        return install_cmd
        
    if (candidate_path / "package.json").exists():
        return "bun install"
    if (candidate_path / "requirements.txt").exists():
        return "pip install -r requirements.txt"
    if (candidate_path / "pyproject.toml").exists():
        return "poetry install"
    if (candidate_path / "go.mod").exists():
        return "go mod tidy"
    if (candidate_path / "Cargo.toml").exists():
        return "cargo fetch"
    return None

def _detect_build_cmd(candidate_path: Path, ctx: dict) -> str | None:
    run_cmds = ctx.get("run_commands", {})
    build_cmd = run_cmds.get("build")
    if build_cmd:
        return build_cmd
        
    if (candidate_path / "package.json").exists():
        # Heuristic: try parsing package.json to see if 'build' script exists
        try:
            with open(candidate_path / "package.json", "r", encoding="utf-8") as f:
                pkg = json.load(f)
                if "scripts" in pkg and "build" in pkg["scripts"]:
                    return "bun run build"
        except Exception:
            pass
            
    return None

def mask_secrets(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'sk-[a-zA-Z0-9]{48}', 'sk-***', text)
    text = re.sub(r'ghp_[a-zA-Z0-9]{36}', 'ghp_***', text)
    return text

def _reader_thread(pipe, out_queue, stream_name, on_log_line):
    for line in iter(pipe.readline, ''):
        if line:
            clean_line = mask_secrets(line.rstrip('\r\n'))
            out_queue.put((stream_name, clean_line))
            if on_log_line:
                try:
                    on_log_line(clean_line)
                except Exception:
                    pass
    pipe.close()

def run_command(
    command: str, 
    cwd: Path, 
    timeout: int, 
    phase: str, 
    on_log_line: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    env = os.environ.copy()
    env["CI"] = "true" 
    
    start_time = time.time()
    if on_log_line:
        on_log_line(f"--- Executing: {command} (cwd: {cwd}) ---")

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1, # Line buffered
            env=env
        )
        
        out_queue = queue.Queue()
        
        stdout_thread = threading.Thread(target=_reader_thread, args=(process.stdout, out_queue, 'stdout', on_log_line))
        stderr_thread = threading.Thread(target=_reader_thread, args=(process.stderr, out_queue, 'stderr', on_log_line))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_thread.join()
            stderr_thread.join()
            
            # Drain queue
            stdout_lines = []
            stderr_lines = []
            while not out_queue.empty():
                stream_name, line = out_queue.get()
                if stream_name == 'stdout':
                    stdout_lines.append(line)
                else:
                    stderr_lines.append(line)
                    
            stderr = "\n".join(stderr_lines) + f"\nProcess timed out after {timeout} seconds."
            if on_log_line:
                on_log_line(f"Process timed out after {timeout} seconds.")
                
            return {
                "passed": False,
                "command": command,
                "phase": phase,
                "exit_code": 124,
                "stdout": "\n".join(stdout_lines),
                "stderr": stderr,
                "timeout_expired": True,
            }
            
        stdout_thread.join()
        stderr_thread.join()
        
        stdout_lines = []
        stderr_lines = []
        while not out_queue.empty():
            stream_name, line = out_queue.get()
            if stream_name == 'stdout':
                stdout_lines.append(line)
            else:
                stderr_lines.append(line)
                
        passed = (process.returncode == 0)
        
        duration = time.time() - start_time
        if on_log_line:
            status = "SUCCESS" if passed else f"FAILED (code {process.returncode})"
            on_log_line(f"--- Finished: {command} in {duration:.1f}s [{status}] ---")

        return {
            "passed": passed,
            "command": command,
            "phase": phase,
            "exit_code": process.returncode,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "timeout_expired": False,
        }
    except Exception as e:
        if on_log_line:
            on_log_line(f"Failed to execute: {str(e)}")
        return {
            "passed": False,
            "command": command,
            "phase": phase,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"Failed to execute: {str(e)}",
            "timeout_expired": False,
        }

def classify_failure(result: Dict[str, Any]) -> str:
    if result["passed"]:
        return "passed"
        
    phase = result["phase"]
    if result["timeout_expired"]:
        return f"{phase}_timeout"

    stderr_lower = result["stderr"].lower()
    stdout_lower = result["stdout"].lower()
    combined_lower = stderr_lower + " " + stdout_lower
    
    # Check for missing tooling
    if "command not found" in stderr_lower or "not recognized as an internal or external command" in stderr_lower:
        return "missing_tooling"

    if phase == "dependency":
        return "dependency_install_failed"
        
    # Check for missing dependencies during build
    missing_dep_signals = [
        "module not found", 
        "no module named", 
        "cannot find module",
        "could not find a version that satisfies the requirement"
    ]
    if any(sig in combined_lower for sig in missing_dep_signals):
        return "missing_dependency"
        
    if "syntaxerror" in stderr_lower or "syntax error" in stderr_lower:
        return "syntax_error"
        
    if "typeerror" in stderr_lower or "type error" in stderr_lower:
        return "type_error"
        
    return "build_failed"

def _format_result(results: list[Dict[str, Any]], language: str, default_phase: str) -> dict:
    if not results:
        return {
            "passed": True,
            "skipped": True,
            "detected_language": language,
            "command": "none",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "category": "passed",
            "error_summary": "Skipped (no commands)",
        }
        
    failed_result = next((r for r in results if not r["passed"]), None)
    
    if failed_result:
        category = classify_failure(failed_result)
        return {
            "passed": False,
            "skipped": False,
            "detected_language": language,
            "command": failed_result["command"],
            "phase": failed_result["phase"],
            "exit_code": failed_result["exit_code"],
            "stdout": failed_result["stdout"][-5000:],
            "stderr": failed_result["stderr"][-5000:],
            "category": category,
            "error_summary": f"{failed_result['phase'].capitalize()} failed: {category}",
        }
        
    # All passed
    last_res = results[-1]
    combined_stdout = "\\n".join(f"--- Executed: {r['command']} ---\\n{r['stdout']}" for r in results)
    combined_stderr = "\\n".join(f"--- Executed: {r['command']} ---\\n{r['stderr']}" for r in results if r['stderr'])
    
    return {
        "passed": True,
        "skipped": False,
        "detected_language": language,
        "command": last_res["command"],
        "phase": last_res["phase"],
        "exit_code": 0,
        "stdout": combined_stdout[-5000:],
        "stderr": combined_stderr[-5000:],
        "category": "passed",
        "error_summary": "",
    }

def get_estimated_install_time(command: str) -> int:
    cmd_lower = command.lower()
    if "bun install" in cmd_lower or "bun create" in cmd_lower:
        return 15
    if "npm ci" in cmd_lower or "npm install" in cmd_lower:
        return 45
    if "uv pip install" in cmd_lower:
        return 10
    if "pip install" in cmd_lower:
        return 30
    if "poetry install" in cmd_lower:
        return 40
    if "go mod" in cmd_lower:
        return 5
    if "cargo fetch" in cmd_lower:
        return 20
    return 30

def ensure_dependencies(candidate_dir: str, project_context_str: str | None, on_log_line: Optional[Callable[[str], None]] = None) -> dict:
    candidate_path = Path(candidate_dir)
    ctx = _parse_project_context(project_context_str)
    language = _detect_language(candidate_path, ctx)
    install_cmd = _detect_install_cmd(candidate_path, ctx)
    
    if not install_cmd:
        return _format_result([], language, "dependency")
        
    res = run_command(install_cmd, candidate_path, 600, "dependency", on_log_line)
    return _format_result([res], language, "dependency")

def run_build(candidate_dir: str, project_context_str: str | None, on_log_line: Optional[Callable[[str], None]] = None) -> dict:
    candidate_path = Path(candidate_dir)
    ctx = _parse_project_context(project_context_str)
    language = _detect_language(candidate_path, ctx)
    build_cmd = _detect_build_cmd(candidate_path, ctx)
    
    if not build_cmd:
        return _format_result([], language, "build")
        
    res = run_command(build_cmd, candidate_path, 180, "build", on_log_line)
    return _format_result([res], language, "build")
