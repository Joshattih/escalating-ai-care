"""
call_model.py  —  one function that sends one prompt to one model and logs the call.

Nothing in this file runs on import. `ask(...)` is the only entry point and it is
called by probe.py and round0.py. Importing this file does not call any API.

What one call does, in order:
  1. build the prompt from the pieces (role text or none, statement, question, format)
  2. send it to the named model with the fixed sampling settings from config.py
  3. retry a few times on a network or rate-limit error, waiting longer each time
  4. parse the reply: the first token must be 1-6 or IDK; the rest is the reason
  5. write one JSON line to the run's log with everything needed to rebuild the
     result later: model, role id, statement id, prompt hash, settings, raw text,
     parsed answer, parse status, time, and how many retries it took

Parse status is one of:
  "valid"    first token is 1, 2, 3, 4, 5 or 6
  "idk"      first token is IDK
  "invalid"  anything else (empty reply, a sentence with no number, a 7, ...)
Invalid replies are logged and counted. They are never turned into a rating or an IDK.

Keys are read from the environment at call time (OPENAI_API_KEY, ANTHROPIC_API_KEY,
GOOGLE_API_KEY) and never written anywhere.
"""

import os, re, json, time, hashlib
from datetime import datetime, timezone
from pathlib import Path

from config import QUESTION, SCALE, IDK, MODELS, TEMPERATURE, MAX_OUTPUT_TOKENS

# ---------------------------------------------------------------------------
# The prompt. Every call uses exactly this shape. The role line is absent in the
# probe and present in round 0. Nothing else differs between the two.
# ---------------------------------------------------------------------------
SCALE_TEXT = "\n".join(f"{k}  {v}" for k, v in SCALE.items())

ANSWER_FORMAT = (
    f"Answer on the first line with exactly one of: 1, 2, 3, 4, 5, 6, or {IDK}.\n"
    "On the second line give one sentence of reason."
)

def build_prompt(statement_text, role_text=None):
    """Return (system_text, user_text). system_text is None when there is no role."""
    user = (
        f"Statement: {statement_text}\n\n"
        f"{QUESTION}\n\n"
        f"Scale:\n{SCALE_TEXT}\n{IDK}  I don't know\n\n"
        f"{ANSWER_FORMAT}"
    )
    return role_text, user

def prompt_hash(system_text, user_text):
    """A short fingerprint of the exact prompt, so the log can prove what was sent."""
    h = hashlib.sha256()
    h.update((system_text or "").encode()); h.update(b"\n---\n"); h.update(user_text.encode())
    return h.hexdigest()[:16]

# ---------------------------------------------------------------------------
# Parsing. Strict on purpose: the first non-empty line must be the answer alone.
# ---------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"^\s*(?:answer\s*[:\-]\s*)?([1-6]|IDK)\b\.?\s*$", re.IGNORECASE)

def parse_reply(raw):
    """Return (answer, reason, status). answer is "1".."6", "IDK", or None."""
    lines = [ln.strip() for ln in (raw or "").strip().splitlines() if ln.strip()]
    if not lines:
        return None, "", "invalid"
    m = _ANSWER_RE.match(lines[0])
    if not m:
        return None, " ".join(lines), "invalid"
    tok = m.group(1).upper()
    reason = " ".join(lines[1:]).strip()
    return tok, reason, ("idk" if tok == IDK else "valid")

# ---------------------------------------------------------------------------
# Provider calls. Each returns the raw text of the reply, nothing else.
# The three gotchas from the project notes are respected here:
#   gpt-4.1 family uses max_completion_tokens (temperature is allowed)
#   Anthropic uses the messages API with max_tokens
#   Google may return multi-part replies where .text raises; read the parts instead
# ---------------------------------------------------------------------------
def _call_openai(model, system_text, user_text):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[MODELS[model]["env_key"]])
    msgs = ([{"role": "system", "content": system_text}] if system_text else []) + [{"role": "user", "content": user_text}]
    r = client.chat.completions.create(model=model, messages=msgs, temperature=TEMPERATURE,
                                       max_completion_tokens=MAX_OUTPUT_TOKENS)
    return r.choices[0].message.content or ""

def _call_anthropic(model, system_text, user_text):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ[MODELS[model]["env_key"]])
    kw = dict(model=model, max_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE,
              messages=[{"role": "user", "content": user_text}])
    if system_text: kw["system"] = system_text
    r = client.messages.create(**kw)
    return "".join(getattr(b, "text", "") for b in r.content)

def _call_google(model, system_text, user_text):
    import google.generativeai as genai
    genai.configure(api_key=os.environ[MODELS[model]["env_key"]])
    m = genai.GenerativeModel(model, system_instruction=system_text) if system_text else genai.GenerativeModel(model)
    r = m.generate_content(user_text, generation_config={"temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS})
    try:
        return r.text or ""
    except ValueError:  # multi-part or blocked reply: read the parts directly
        try:
            return "".join(p.text for p in r.candidates[0].content.parts if hasattr(p, "text"))
        except Exception:
            return ""

_CALLERS = {"openai": _call_openai, "anthropic": _call_anthropic, "google": _call_google}

# ---------------------------------------------------------------------------
# The one entry point.
# ---------------------------------------------------------------------------
def ask(model, statement_id, statement_text, role_id=None, role_text=None,
        run_dir=None, repeat=None, round_id=None, retries=4):
    """
    Send one prompt to one model, parse the reply, log one JSON line, return the record.

    model          one of the keys in config.MODELS
    statement_id   e.g. "03-032"; statement_text its sentence
    role_id/text   None for the probe; "D4" and its three lines for round 0
    run_dir        folder for this run's log (created if missing); None = no log
    repeat         probe repeat number 1..20, or None
    round_id       "probe", "round0", ... written into the log
    """
    system_text, user_text = build_prompt(statement_text, role_text)
    rec = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "round": round_id, "model": model, "role_id": role_id, "statement_id": statement_id,
        "repeat": repeat, "prompt_hash": prompt_hash(system_text, user_text),
        "temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "raw": None, "answer": None, "reason": None, "status": None, "attempts": 0, "error": None,
    }
    caller = _CALLERS[MODELS[model]["provider"]]
    wait = 2.0
    for attempt in range(1, retries + 1):
        rec["attempts"] = attempt
        try:
            raw = caller(model, system_text, user_text)
            rec["raw"] = raw
            rec["answer"], rec["reason"], rec["status"] = parse_reply(raw)
            rec["error"] = None
            break
        except Exception as e:           # network, rate limit, provider error: wait and retry
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["status"] = "error"
            time.sleep(wait); wait *= 2
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(run_dir) / "calls.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

if __name__ == "__main__":
    # Dry demonstration: build one prompt and show it. No API call is made here.
    from roles import ROLE_TEXTS
    sysm, usr = build_prompt("At some point in the past month, not necessarily today, the patient had suicidal thoughts and some intention of acting on them.", ROLE_TEXTS["D4"])
    print("=== system (role) ===\n" + sysm + "\n\n=== user ===\n" + usr)
    print("\nparse test:", parse_reply("6\nIntention within a month means a person must assess."), parse_reply("IDK\nNot enough said."), parse_reply("I think 5 is right."))
