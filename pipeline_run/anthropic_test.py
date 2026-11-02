"""Tiny connection test for Sonnet 5.5 through the Anthropic API directly (one short request, a few tokens).

The key comes from the environment variable ANTHROPIC_API_KEY, never from the code.

Run:  python anthropic_test.py
"""
import os
import sys

try:
    import anthropic
except ImportError:
    sys.exit("The 'anthropic' package is not installed. Run: python -m pip install -U anthropic")

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("Set ANTHROPIC_API_KEY first (see the commands given in chat).")

MODEL = "claude-sonnet-5-5"  # same model as llm_claude.py

client = anthropic.Anthropic()
try:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Rate the fame of the film director Christopher Nolan from 1 (unknown) "
                                              "to 10 (world famous). Reply with just the number."}],
    )
except anthropic.AuthenticationError:
    sys.exit("The key was rejected (401). Check that ANTHROPIC_API_KEY is copied completely.")
except anthropic.PermissionDeniedError:
    sys.exit("The key has no access (403). Check the workspace or region settings in the Console.")
except anthropic.RateLimitError:
    sys.exit("Rate limit or no credit (429). Check Billing in the Console.")
except anthropic.APIStatusError as e:
    sys.exit(f"API error {e.status_code}: {e.message}")
except anthropic.APIConnectionError:
    sys.exit("No connection to api.anthropic.com. Check the internet connection or a VPN/proxy.")

text = next((b.text for b in resp.content if b.type == "text"), "")
print("answer      :", text.strip())
print("model       :", resp.model)
print("stop_reason :", resp.stop_reason)
print("tokens      :", resp.usage.input_tokens, "in /", resp.usage.output_tokens, "out")
print("OK - the key works.")
