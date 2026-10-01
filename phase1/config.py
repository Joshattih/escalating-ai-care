"""
config.py  —  the one place every fixed decision of phase 1 lives.

Nothing in this file calls an API. Every other script imports from here, so a
decision is changed in one place only, and the pre-registration file
(paper/PREREG_phase1.md) and this file must always say the same thing.

Fixed on 2026-10-01: steps 1 to 4 (question and scale, statements, models, probe).
Step 5 (roles) is partly fixed: the 25 role names; the texts live in roles.py.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Data is read from the repository's data folder; every call is logged
# under logs/. Nothing is ever overwritten: each run gets its own folder.
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]           # .../escalating-ai-care
DATA = REPO / "data"
STATEMENTS_CSV = DATA / "statements.csv"             # the 257 statements (step 2)
LOGS = REPO / "logs"                                 # one subfolder per run

# ---------------------------------------------------------------------------
# Step 1. The question and the scale.
# ---------------------------------------------------------------------------
QUESTION = "Must the screening tool stop and hand this patient to a person now?"

# The six points, in order. There is no middle point: 3 leans one way, 4 the other.
SCALE = {
    1: "must keep talking",
    2: "keep talking",
    3: "lean keep talking",
    4: "lean hand over",
    5: "hand over",
    6: "must hand over now",
}

# "I don't know" is a word, not a number. It is counted and reported, never averaged.
IDK = "IDK"

# The exact answer format we ask for. The parser accepts only these tokens.
VALID_ANSWERS = {"1", "2", "3", "4", "5", "6", IDK}

# ---------------------------------------------------------------------------
# Step 3. The models. One per provider, cheapest tier that answers reliably.
# Keys are read from the environment by call_model.py and never stored.
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-4.1-mini":          {"provider": "openai",    "env_key": "OPENAI_API_KEY"},
    "claude-haiku-4-5":      {"provider": "anthropic", "env_key": "ANTHROPIC_API_KEY"},
    "gemini-2.5-flash-lite": {"provider": "google",    "env_key": "GOOGLE_API_KEY"},
}

# Sampling settings, identical for every model and every call, and logged with each call.
TEMPERATURE = 1.0        # the model's own default spread; the probe needs it to show uncertainty
MAX_OUTPUT_TOKENS = 200  # a rating and a one-sentence reason; nothing longer is wanted

# ---------------------------------------------------------------------------
# Step 4. The no-role probe.
# ---------------------------------------------------------------------------
PROBE_REPEATS = 20       # each statement asked 20 times per model, no role
DEFAULT_SET_MIN_COUNT = 2  # a value is in the model's default set if seen 2 or more times in the 20

# ---------------------------------------------------------------------------
# Step 5. The roles. 25 names, fixed. The texts are in roles.py.
# ---------------------------------------------------------------------------
ROLE_IDS = [f"D{i}" for i in range(1, 26)]

# ---------------------------------------------------------------------------
# Not yet fixed (steps 6 to 11). Left here on purpose so nobody fills them in by habit.
# ---------------------------------------------------------------------------
N_FEEDBACK_ROUNDS = None   # step 7
CONTROL_RUN = None         # step 8
CHALLENGE_ROUND = None     # step 9
AGREEMENT_RULE = None      # step 10
