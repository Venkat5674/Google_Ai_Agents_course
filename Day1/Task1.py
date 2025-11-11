#1 — Install (optional)
# run in terminal / notebook cell if required
pip install google-adk google-genai kaggle jupyter-server


#2 — Configure Gemini API Key (Kaggle secrets)
# Kaggle users: retrieve secret and set env var
import os
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print("🔑 Authentication Error: Please add 'GOOGLE_API_KEY' to your Kaggle secrets. Details:", e)

#3 — Imports (ADK + helpers)
# core ADK imports
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

# stdlib & environment
import os
import asyncio
import subprocess
from IPython.core.display import display, HTML


# 4 — Helper: get_adk_proxy_url (Kaggle Notebook / Jupyter)
# helper to produce the proxied URL and display a clickable button (Kaggle-style)
def get_adk_proxy_url():
    # NOTE: this function expects to run inside a Kaggle Notebook/Jupyter env where
    # jupyter_server.serverapp.list_running_servers() returns a base_url with kernel/token.
    try:
        from jupyter_server.serverapp import list_running_servers
    except Exception as e:
        raise RuntimeError("jupyter_server import failed. This helper is intended for Kaggle/Jupyter environments.") from e

    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found. Cannot build proxy URL.")

    baseURL = servers[0]["base_url"]
    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>⚠️ IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one that starts <code>adk web</code>).</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before starting the web server, you may get an error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) ↗
        </a>
    </div>
    """
    display(HTML(styled_html))
    return url_prefix

# Quick check
print("✅ Helper functions ready.")

#5 — Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5,        # Maximum retry attempts
    exp_base=7,        # Delay multiplier
    initial_delay=1,   # Initial delay before first retry (seconds)
    http_status_codes=[429, 500, 503, 504]
)
print("✅ Retry config set.")

#6 — Define the Agent
root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

print("✅ Root Agent defined.")

#7 — Create Runner & Run queries (use asyncio.run in notebook)
runner = InMemoryRunner(agent=root_agent)
print("✅ Runner created.")

# async wrapper to call run_debug
async def ask_agent(prompt: str):
    response = await runner.run_debug(prompt)
    # response is typically a structure; printing the assistant reply text is common:
    # ADK's run_debug prints trace; many examples return text in response.result or response[...]
    # For the notebook trace, you can inspect the object directly.
    print("\n--- Agent Response Trace ---\n")
    print(response)  # inspect the returned object in your environment

# Examples: run two example prompts sequentially
async def run_examples():
    await ask_agent("What is Agent Development Kit from Google? What languages is the SDK available in?")
    await ask_agent("What's the weather in London?")

# execute
asyncio.run(run_examples())

# 8 — Create a sample-agent folder (shell) from Python (same as !adk create ...)
# Use subprocess to run the adk CLI create command (this creates sample-agent folder)
import shlex, subprocess

create_cmd = f"adk create sample-agent --model gemini-2.5-flash-lite --api_key {os.environ.get('GOOGLE_API_KEY','')}"
print("Running:", create_cmd)
proc = subprocess.run(shlex.split(create_cmd), capture_output=True, text=True)
print(proc.stdout)
if proc.stderr:
    print("STDERR:", proc.stderr)
  
#9 — Get proxy URL and start ADK web (non-blocking)
# Build proxy url prefix (Kaggle/Jupyter only)
try:
    url_prefix = get_adk_proxy_url()
    print("URL prefix:", url_prefix)
except Exception as e:
    print("Could not build proxy URL (not in Kaggle/Jupyter?):", e)
    url_prefix = None

# Start ADK web server non-blocking (will run until stopped). This launches ADK's web UI.
# NOTE: This command will run the web server and stream logs to stdout - usually you run it in its own terminal.
if url_prefix:
    web_cmd = ["adk", "web", "--url_prefix", url_prefix]
    print("Starting ADK web UI with:", " ".join(web_cmd))
    # Use Popen so the notebook cell doesn't block. Logs will still appear in the kernel output.
    web_proc = subprocess.Popen(web_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"ADK web started (pid={web_proc.pid}). Check kernel output for logs. Open the proxy link displayed above.")
else:
    print("ADK web not started (missing url_prefix). If running locally, run: adk web --url_prefix <your_prefix> or simply `adk web`")
