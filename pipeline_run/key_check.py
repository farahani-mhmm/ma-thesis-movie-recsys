"""Find out which provider an API key belongs to and which models it can use. Nothing is generated, so it costs nothing.

The key is typed hidden (it does not appear on the screen) or read from the environment variable TEST_API_KEY.
It is never printed; only its first 4 characters and its length are shown.

The provider is guessed from how the key starts. If the start is not recognised, the key is sent to the providers
one by one only when you add --try-all (each of them then sees the key).

Run (from cold_start_lab):
    .venv\\Scripts\\python.exe ..\\pipeline_run\\key_check.py
    .venv\\Scripts\\python.exe ..\\pipeline_run\\key_check.py --try-all
"""
import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request

# OpenAI-style providers: GET <base>/models with "Authorization: Bearer <key>"
OPENAI_STYLE = {
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Groq": "https://api.groq.com/openai/v1",
    "xAI (Grok)": "https://api.x.ai/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "Mistral": "https://api.mistral.ai/v1",
    "AI/ML API": "https://api.aimlapi.com/v1",     # aimlapi.com; its keys are plain hex, so pass --provider aimlapi
}
PROVIDER_ALIASES = {"anthropic": "Anthropic", "gemini": "Google Gemini", "openai": "OpenAI", "openrouter": "OpenRouter",
                    "groq": "Groq", "xai": "xAI (Grok)", "deepseek": "DeepSeek", "mistral": "Mistral", "aimlapi": "AI/ML API"}

# (prefix, provider), longest prefixes first
PREFIXES = [
    ("sk-ant-", "Anthropic"), ("sk-or-", "OpenRouter"), ("sk-proj-", "OpenAI"), ("sk-svcacct-", "OpenAI"),
    ("gsk_", "Groq"), ("xai-", "xAI (Grok)"), ("AIza", "Google Gemini"), ("AQ.", "Google Gemini"), ("sk-", "OpenAI"),
]


def guess(key):
    return next((p for pre, p in PREFIXES if key.startswith(pre)), None)


def http_json(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def check_openai_style(name, key):
    data = http_json(OPENAI_STYLE[name] + "/models", {"Authorization": f"Bearer {key}"})
    items = data if isinstance(data, list) else data.get("data", [])
    return sorted(m.get("id", "?") if isinstance(m, dict) else str(m) for m in items)


def check_anthropic(key):
    try:
        import anthropic
    except ImportError:
        sys.exit("Run first: .venv\\Scripts\\python.exe -m pip install -U anthropic")
    client = anthropic.Anthropic(api_key=key)
    return sorted(m.id for m in client.models.list())


def check_gemini(key):
    try:
        from google import genai
    except ImportError:
        sys.exit("Run first: .venv\\Scripts\\python.exe -m pip install -U google-genai")
    client = genai.Client(api_key=key)
    return sorted(m.name.removeprefix("models/") for m in client.models.list()
                  if "generateContent" in (m.supported_actions or []))


def check(provider, key):
    if provider == "Anthropic":
        return check_anthropic(key)
    if provider == "Google Gemini":
        return check_gemini(key)
    return check_openai_style(provider, key)


def explain(e):
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code in (401, 403) or "API key not valid" in str(e):
        return f"rejected ({code}): the key is not valid for this provider"
    if isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError):
        return f"no connection ({e.reason})"
    return f"{type(e).__name__}: {str(e)[:200]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--try-all", action="store_true", help="if the provider is not recognised, try every provider")
    ap.add_argument("--provider", choices=sorted(PROVIDER_ALIASES), help="skip the guess and test this provider only")
    a = ap.parse_args()

    key = os.environ.get("TEST_API_KEY", "").strip() or getpass.getpass("Paste the key (it stays hidden), then Enter: ").strip()
    if not key:
        sys.exit("No key given.")
    print(f"key starts with: {key[:4]}...   length: {len(key)}")

    provider = PROVIDER_ALIASES[a.provider] if a.provider else guess(key)
    if provider:
        candidates = [provider]
        print(f"looks like: {provider}")
    elif a.try_all:
        candidates = ["Anthropic", "Google Gemini", *OPENAI_STYLE]
        print("provider not recognised from the start of the key; trying all (each provider will see the key)")
    else:
        sys.exit("Provider not recognised from the start of the key. Add --try-all to test it against every provider, "
                 "or tell me where the key came from (for Liara the base address of the panel is needed too).")

    for prov in candidates:
        try:
            models = check(prov, key)
        except Exception as e:
            print(f"\n{prov}: {explain(e)}")
            continue
        print(f"\n{prov}: the key WORKS. Models it can use ({len(models)}):")
        for m in models:
            print("   ", m)
        return
    print("\nThe key did not work with any provider that was tried.")


if __name__ == "__main__":
    main()
