"""One subprocess streaming/logger implementation for scripts and notebook clients."""
from collections import deque
import os
from pathlib import Path
import subprocess

from .config import PROJECT_ROOT


def run_logged(name: str, argv: list[str], log_handle=None, *, log_path: Path | None = None,
               cwd: Path = PROJECT_ROOT, live: bool = True) -> int:
    if log_path is not None:
        with Path(log_path).open("w", encoding="utf-8", buffering=1) as stream:
            return run_logged(name, argv, stream, cwd=cwd, live=live)
    print(f"CHECK: {name}\nCOMMAND: {subprocess.list2cmdline([str(x) for x in argv])}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    tail = deque(maxlen=12)
    try:
        with subprocess.Popen([str(x) for x in argv], cwd=cwd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                if log_handle is not None:
                    log_handle.write(line)
                    log_handle.flush()
                if live:
                    print(line, end="", flush=True)
                else:
                    tail.append(line.rstrip())
            code = process.wait()
    except OSError as exc:
        message = f"Could not start command: {exc}"
        print(message)
        if log_handle is not None:
            log_handle.write(message + "\n")
        code = 127
    if not live:
        for line in tail:
            print(line)
    print(f"RESULT: {'PASS' if code == 0 else 'FAIL'} ({name}); exit={code}", flush=True)
    if log_handle is not None:
        log_handle.write(f"EXIT CODE: {code}\n")
        log_handle.flush()
    return code
