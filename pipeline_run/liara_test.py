"""Tiny connection test for Sonnet 5.5 through Liara's OpenAI-compatible endpoint (one short request, a few tokens).

Settings come from environment variables, never from the code:
    LIARA_BASE_URL   e.g. https://ai.liara.ir/api/<your id>/v1
    LIARA_API_KEY    the access key made in the Liara panel
    LIARA_MODEL      optional, default anthropic/claude-sonnet-5.5

Run:  python liara_test.py
"""
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("The 'openai' package is not installed. Run: python -m pip install -U openai")

base_url = os.environ.get("LIARA_BASE_URL")
api_key = os.environ.get("LIARA_API_KEY")
model = os.environ.get("LIARA_MODEL", "anthropic/claude-sonnet-5.5")
if not base_url or not api_key:
    sys.exit("Set LIARA_BASE_URL and LIARA_API_KEY first (see the commands given in chat).")

client = OpenAI(base_url=base_url, api_key=api_key)
resp = client.chat.completions.create(
    model=model,
    max_tokens=60,
    messages=[{"role": "user", "content": "Rate the fame of the film director Christopher Nolan from 1 (unknown) "
                                          "to 10 (world famous). Reply with just the number."}],
)
print("model :", resp.model)
print("answer:", resp.choices[0].message.content)
print("tokens:", resp.usage)
