"""Run any command and follow its output in a Telegram chat (start, progress every few minutes, errors, end).

The command itself is not changed: its output still appears in the terminal as usual. Only the Python standard
library is used, and a Telegram problem (no network, wrong token) never stops the command.

Settings come from environment variables, never from the code:
    TELEGRAM_BOT_TOKEN   the token BotFather gives you
    TELEGRAM_CHAT_ID     your chat id (find it with --find-chat-id after sending any message to your bot)

Run (from cold_start_lab):
    python tg_run.py --find-chat-id                       print the chat ids that recently wrote to the bot
    python tg_run.py --test                               send one test message
    python tg_run.py --name gemini37 -- .venv\\Scripts\\python.exe ..\\pipeline_run\\llm_gemini.py --model gemini-3.7-flash --yes
    python tg_run.py --dry -- python -c "print('hi')"     print the messages here instead of sending them

Options: --every SECONDS between progress messages (default 600), --tail LINES per message (default 15).
"""
import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime

API = "https://api.telegram.org/bot{token}/{method}"
MAX_CHARS = 3800                    # Telegram allows 4096 per message; leave room for the header
ALERT_GAP_SECONDS = 60              # at most one error alert per minute
ALERT_PATTERN = re.compile(r"Traceback|Error|error \d|\b(429|500|503)\b|stopping|Not finished|Stopped", re.IGNORECASE)


class Telegram:
    def __init__(self, dry):
        self.dry = dry
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    def call(self, method, params):
        url = API.format(token=self.token, method=method)
        data = urllib.parse.urlencode(params).encode("utf-8")
        with urllib.request.urlopen(url, data=data, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))

    def send(self, title, lines=()):
        body = "\n".join(lines)
        if len(body) > MAX_CHARS:
            body = "...\n" + body[-MAX_CHARS:]
        if self.dry:
            print(f"\n[telegram, dry] {title}\n{body}\n", flush=True)
            return
        text = f"<b>{html.escape(title)}</b>" + (f"\n<pre>{html.escape(body)}</pre>" if body else "")
        try:
            self.call("sendMessage", {"chat_id": self.chat, "text": text, "parse_mode": "HTML",
                                      "disable_web_page_preview": "true"})
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            print(f"[tg_run] could not send to Telegram ({type(e).__name__}); the command keeps running", flush=True)


def find_chat_id(tg):
    try:
        res = tg.call("getUpdates", {"limit": 20})
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"Could not reach Telegram: {e}")
    chats = {}
    for upd in res.get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        c = msg.get("chat", {})
        if c:
            chats[c["id"]] = c.get("username") or c.get("title") or c.get("first_name") or ""
    if not chats:
        sys.exit("No messages found. Send any message to your bot in Telegram, then run this again.")
    for cid, who in chats.items():
        print(f"chat id: {cid}   ({who})")


def fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="short label shown in every message")
    ap.add_argument("--every", type=int, default=600, help="seconds between progress messages")
    ap.add_argument("--tail", type=int, default=15, help="output lines per message")
    ap.add_argument("--dry", action="store_true", help="print the messages instead of sending them")
    ap.add_argument("--find-chat-id", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="the command to run, after --")
    a = ap.parse_args()

    tg = Telegram(a.dry)
    if not a.dry and not tg.token:
        sys.exit("Set TELEGRAM_BOT_TOKEN first (see the commands given in chat).")
    if a.find_chat_id:
        find_chat_id(tg)
        return
    if not a.dry and not tg.chat:
        sys.exit("Set TELEGRAM_CHAT_ID first. Find it with: python tg_run.py --find-chat-id")
    if a.test:
        tg.send("tg_run test: the bot can reach you", [datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        print("Test message sent (check Telegram).")
        return

    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        sys.exit("Give the command to run after --")
    label = a.name or os.path.basename(cmd[1] if len(cmd) > 1 else cmd[0])   # usually the script after python.exe
    tail = deque(maxlen=a.tail)
    lock = threading.Lock()
    done = threading.Event()
    start = time.time()
    last_alert = [0.0]

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    tg.send(f"[{label}] started", [" ".join(cmd)])
    exe = os.path.abspath(cmd[0]) if os.path.isfile(cmd[0]) else shutil.which(cmd[0])   # Windows needs a full path here
    if not exe:
        tg.send(f"[{label}] could not start: program not found", [cmd[0]])
        sys.exit(f"Program not found: {cmd[0]}")
    proc = subprocess.Popen([exe] + cmd[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=1,
                            text=True, encoding="utf-8", errors="replace")

    def progress():
        while not done.wait(a.every):
            with lock:
                lines = list(tail)
            tg.send(f"[{label}] running, {fmt_duration(time.time() - start)}", lines)

    threading.Thread(target=progress, daemon=True).start()

    interrupted = False
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            line = line.rstrip("\n")
            with lock:
                tail.append(line)
                lines = list(tail)
            if ALERT_PATTERN.search(line) and time.time() - last_alert[0] > ALERT_GAP_SECONDS:
                last_alert[0] = time.time()
                tg.send(f"[{label}] warning", lines[-5:])
        proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        done.set()

    with lock:
        lines = list(tail)
    took = fmt_duration(time.time() - start)
    if interrupted:
        tg.send(f"[{label}] stopped by you (Ctrl+C) after {took}", lines)
    elif proc.returncode == 0:
        tg.send(f"[{label}] finished OK in {took}", lines)
    else:
        tg.send(f"[{label}] ended with exit code {proc.returncode} after {took}", lines)
    sys.exit(130 if interrupted else proc.returncode)


if __name__ == "__main__":
    main()
