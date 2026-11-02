"""Tiny connection test for the Gemini API (lists the models the key can use, then one short request, a few tokens).

Settings come from environment variables, never from the code:
    GEMINI_API_KEY   the key made in Google AI Studio
    GEMINI_MODEL     optional; if not set, gemini-flash-latest (or the first "flash" model the key can use) is taken

Run:  python gemini_test.py
"""
import os
import sys

try:
    from google import genai
    from google.genai import errors
except ImportError:
    sys.exit("The 'google-genai' package is not installed. Run: python -m pip install -U google-genai")

if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    sys.exit("Set GEMINI_API_KEY first (see the commands given in chat).")

client = genai.Client()  # key from GEMINI_API_KEY (or GOOGLE_API_KEY)

try:
    names = [m.name.removeprefix("models/") for m in client.models.list()
             if "generateContent" in (m.supported_actions or [])]
except errors.APIError as e:
    sys.exit(f"The key was rejected or has no access ({e.code}): {e.message}")
except Exception as e:  # network problems, proxy, DNS
    sys.exit(f"No connection to the Gemini API: {e}")

print(f"models this key can use for text ({len(names)}):")
for n in names:
    print("   ", n)

# "gemini-flash-latest" always points at the current flash model; older ids may be closed to new users
model = os.environ.get("GEMINI_MODEL") or ("gemini-flash-latest" if "gemini-flash-latest" in names else
                                           next((n for n in names if "flash" in n and "lite" not in n), names[0] if names else None))
if not model:
    sys.exit("The key lists no text model.")

try:
    resp = client.models.generate_content(
        model=model,
        contents="Rate the fame of the film director Christopher Nolan from 1 (unknown) to 10 (world famous). "
                 "Reply with just the number.",
    )
except errors.APIError as e:
    sys.exit(f"API error {e.code} with model {model}: {e.message}")

usage = resp.usage_metadata
print("model       :", model)
print("answer      :", (resp.text or "").strip())
if usage:
    print("tokens      :", usage.prompt_token_count, "in /", usage.candidates_token_count, "out")
print("OK - the key works.")
