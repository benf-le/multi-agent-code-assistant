import json
import os
import subprocess
from pathlib import Path

def run_build_test(candidate_dir: str, project_context_str: str | None) -> dict:
    """Run build and test commands on the candidate project.

    Uses commands defined in project_context.json if available,
    otherwise attempts basic fallbacks.
    
    Timeout: 180 seconds.
    """
    candidate_path = Path(candidate_dir)
    
    # 1. Parse project context for commands
    install_cmd = None
    build_cmd = None
    test_cmd = None
    language = "unknown"
    
    if project_context_str:
        try:
            # We assume extract_project_context_from_current_project format:
            # ## project_context.json
            # { ... }
            # Let's extract the JSON part.
            json_str = project_context_str
            if "## project_context.json" in json_str:
                parts = json_str.split("## project_context.json")
                if len(parts) > 1:
                    json_str = parts[1].strip()
            
            # Simple heuristic if there's markdown code blocks
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
                
            ctx = json.loads(json_str)
            language = ctx.get("stack", {}).get("language", "unknown")
            run_cmds = ctx.get("run_commands", {})
            install_cmd = run_cmds.get("install")
            build_cmd = run_cmds.get("build")  # Sometimes separate
            
            test_cmds_val = ctx.get("test_commands")
            if isinstance(test_cmds_val, list) and test_cmds_val:
                test_cmd = test_cmds_val[0]
            elif isinstance(test_cmds_val, str):
                test_cmd = test_cmds_val
                
        except Exception:
            pass
            
    # 2. Fallbacks if not detected
    if not test_cmd:
        if (candidate_path / "package.json").exists():
            language = "Node.js"
            install_cmd = install_cmd or "npm install"
            test_cmd = "npm test"
        elif (candidate_path / "requirements.txt").exists() or (candidate_path / "pytest.ini").exists():
            language = "Python"
            install_cmd = install_cmd or "pip install -r requirements.txt"
            test_cmd = "pytest"
        elif (candidate_path / "go.mod").exists():
            language = "Go"
            test_cmd = "go test ./..."
        elif list(candidate_path.glob("*.csproj")) or list(candidate_path.glob("*.sln")):
            language = ".NET"
            test_cmd = "dotnet test"
        elif (candidate_path / "pom.xml").exists():
            language = "Java"
            test_cmd = "mvn test"
        elif (candidate_path / "Cargo.toml").exists():
            language = "Rust"
            test_cmd = "cargo test"
            
    # 3. Execution
    commands_to_run = []
    if install_cmd:
        commands_to_run.append(install_cmd)
    if build_cmd:
        commands_to_run.append(build_cmd)
    if test_cmd:
        commands_to_run.append(test_cmd)
        
    if not commands_to_run:
        return {
            "passed": False,
            "detected_language": language,
            "command": "none",
            "exit_code": 1,
            "stdout": "",
            "stderr": "No build/test command found in project_context.json and no fallback matched.",
            "error_summary": "Missing build/test commands",
        }
        
    combined_stdout = ""
    combined_stderr = ""
    last_exit_code = 0
    failed_cmd = ""
    
    env = os.environ.copy()
    # Add non-interactive flags where possible
    env["CI"] = "true" 
    
    for cmd in commands_to_run:
        try:
            # We use shell=True for convenience with combined commands like `npm install && npm run build`
            process = subprocess.run(
                cmd,
                shell=True,
                cwd=candidate_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
                env=env
            )
            combined_stdout += f"\\n--- Executed: {cmd} ---\\n{process.stdout}"
            if process.stderr:
                combined_stderr += f"\\n--- Executed: {cmd} ---\\n{process.stderr}"
            
            last_exit_code = process.returncode
            if last_exit_code != 0:
                failed_cmd = cmd
                break
        except subprocess.TimeoutExpired as e:
            last_exit_code = 124 # Common timeout exit code
            failed_cmd = cmd
            if e.stdout:
                if isinstance(e.stdout, bytes):
                    combined_stdout += f"\\n--- Executed: {cmd} (TIMEOUT) ---\\n{e.stdout.decode('utf-8', errors='replace')}"
                else:
                    combined_stdout += f"\\n--- Executed: {cmd} (TIMEOUT) ---\\n{e.stdout}"
            if e.stderr:
                if isinstance(e.stderr, bytes):
                    combined_stderr += f"\\n--- Executed: {cmd} (TIMEOUT) ---\\n{e.stderr.decode('utf-8', errors='replace')}"
                else:
                    combined_stderr += f"\\n--- Executed: {cmd} (TIMEOUT) ---\\n{e.stderr}"
            combined_stderr += "\\nProcess timed out after 180 seconds."
            break
        except Exception as e:
            last_exit_code = 1
            failed_cmd = cmd
            combined_stderr += f"\\n--- Executed: {cmd} (ERROR) ---\\nFailed to execute: {str(e)}"
            break

    passed = (last_exit_code == 0)
    
    # Try to extract common error summaries
    error_summary = ""
    if not passed:
        stderr_lower = combined_stderr.lower()
        stdout_lower = combined_stdout.lower()
        
        if "module not found" in stderr_lower or "no module named" in stderr_lower or "cannot find module" in stderr_lower:
            error_summary = "Missing imports or modules"
        elif "syntaxerror" in stderr_lower or "syntax error" in stderr_lower:
            error_summary = "Syntax error"
        elif "timeout" in stderr_lower or last_exit_code == 124:
            error_summary = "Command timed out (exceeded 180s)"
        elif "failed" in stdout_lower or "error" in stdout_lower:
            error_summary = "Tests failed or build error"
        else:
            error_summary = "Command returned non-zero exit code"

    return {
        "passed": passed,
        "detected_language": language,
        "command": failed_cmd if not passed else commands_to_run[-1],
        "exit_code": last_exit_code,
        "stdout": combined_stdout.strip()[-5000:], # keep last 5000 chars to avoid massive logs
        "stderr": combined_stderr.strip()[-5000:],
        "error_summary": error_summary,
    }
