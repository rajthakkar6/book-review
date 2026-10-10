"""Every tunable in one place."""
import os
from pathlib import Path



def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines). Real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export "), v.strip().strip("\"'"))


_load_dotenv()

DATA_DIR = Path("data")
LEDGER_PATH = DATA_DIR / "ledger.db"
LABELS_PATH = DATA_DIR / "labels.json"  # ground truth; agent code must never read this

# Money is integer paise. 1 rupee = 100 paise.
APPROVAL_LIMIT_PAISE = 5_000_000_00  # Rs 5,00,000

# Detector thresholds
GST_TOLERANCE_PAISE = 1
SPLIT_WINDOW_DAYS = 3
SPLIT_MIN_FRACTION_PCT = 40  # each split payment >= 40% of the limit
OUTLIER_MIN_HISTORY = 8
OUTLIER_MAD_K = 6  # in scaled MADs

# Tool layer
MAX_LIMIT = 50

# Agent budgets
MAX_STEPS = 12
MAX_TOOL_CALLS = 30
MAX_TOKENS = 60_000
MAX_WALL_SECONDS = 120
MAX_TOOL_OUTPUT_CHARS = 8_000
MAX_REPAIRS = 1

# Model. UNVERIFIED id: "luna-6" is the user's chosen model; confirm the exact API id with the provider.
# Override without code changes: BOOKSREVIEW_MODEL=<id>. For a non-OpenAI endpoint that speaks the
# OpenAI chat API, set OPENAI_BASE_URL (read by the openai SDK) and OPENAI_API_KEY.
MODEL_NAME = os.environ.get("BOOKSREVIEW_MODEL", "luna-6")
# Sent only when set; non-reasoning models (e.g. gpt-4o-mini) reject the parameter. gpt-6-luna needs "none"
# for function tools on chat.completions (set BOOKSREVIEW_REASONING_EFFORT=none).
REASONING_EFFORT = os.environ.get("BOOKSREVIEW_REASONING_EFFORT", "")

# Tracing: full request/response log per run, for manual review. Set BOOKSREVIEW_TRACE=0 to disable.
TRACE_ENABLED = os.environ.get("BOOKSREVIEW_TRACE", "1") != "0"
TRACE_DIR = Path(os.environ.get("BOOKSREVIEW_TRACE_DIR", "logs"))

# Optional LLM judge for action quality in evals. Prefer a different model than the agent's to avoid self-preference.
JUDGE_MODEL = os.environ.get("BOOKSREVIEW_JUDGE_MODEL", MODEL_NAME)

# "Ask" chat: a smaller budget than a full review
ASK_MAX_STEPS = 8
ASK_MAX_TOOL_CALLS = 12
ASK_MAX_TOKENS = 40_000
ASK_MAX_WALL_SECONDS = 90
ASK_MAX_HISTORY_MESSAGES = 12
ASK_MAX_QUESTION_CHARS = 1000
