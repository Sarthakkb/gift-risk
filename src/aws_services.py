"""AWS integrations for GIFT Risk: S3 data loading, Bedrock risk commentary.

Every function degrades gracefully when credentials are absent so the demo
never hard-fails: S3 falls back to local CSVs; Bedrock falls back to a
clearly-labelled offline template.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUCKET = os.getenv("S3_BUCKET_NAME", "gift-risk-hackathon")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
# Claude Opus 5 via the us. cross-region inference profile — the bare model
# ID (anthropic.claude-opus-5) is INFERENCE_PROFILE-only on Bedrock
BEDROCK_MODEL_ID = "us.anthropic.claude-opus-5"


# ---------------------------------------------------------------------- S3
def load_position_csv(pair: str) -> tuple[pd.DataFrame, str]:
    """Load a position's return series from S3, falling back to local disk.

    Returns (dataframe, source) where source is 's3' or 'local'.
    """
    slug = pair.replace("/", "_").lower()
    key = f"data/{slug}.csv"
    try:
        import boto3

        s3 = boto3.client("s3", region_name=REGION)
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        return df, "s3"
    except Exception:
        df = pd.read_csv(DATA_DIR / f"{slug}.csv")
        return df, "local"


def load_metadata() -> tuple[dict, str]:
    """Load positions metadata JSON from S3 with local fallback."""
    try:
        import boto3

        s3 = boto3.client("s3", region_name=REGION)
        obj = s3.get_object(Bucket=BUCKET, Key="data/positions_metadata.json")
        return json.loads(obj["Body"].read()), "s3"
    except Exception:
        return json.loads((DATA_DIR / "positions_metadata.json").read_text()), "local"


# ------------------------------------------------------------------ Bedrock
_OFFLINE_TEMPLATE = (
    "[offline commentary — Bedrock unavailable] Under the '{scenario}' scenario, "
    "the {pair} book shows a 95% one-day VaR of {var_pct:.2%} "
    "({var_usd}). Classical and quantum estimates {agree_txt}. "
    "Quantum amplitude estimation reached this precision with {speedup:.0f}× "
    "fewer samples than classical Monte Carlo at the benchmark target error."
)


def generate_risk_commentary(
    position_name: str,
    pair: str,
    scenario_name: str,
    scenario_narrative: str,
    var_estimate: float,
    var_usd: str,
    classical_quantum_agree: bool,
    speedup: float,
) -> tuple[str, str]:
    """Get 2-3 sentence plain-English risk commentary from Bedrock Claude.

    Returns (text, source) where source is 'bedrock' or 'offline'.
    """
    agree_txt = (
        "agree within tolerance" if classical_quantum_agree else "show divergence"
    )
    prompt = (
        "You are a treasury risk analyst at a GIFT IFSC (GIFT City, India) "
        "banking unit, writing one paragraph for the morning risk report.\n\n"
        f"Position: {position_name} ({pair})\n"
        f"Stress scenario: {scenario_name}\n"
        f"Scenario context: {scenario_narrative}\n"
        f"95% one-day VaR estimate: {var_estimate:.2%} of notional ({var_usd})\n"
        f"Model check: classical Monte Carlo and quantum amplitude estimation {agree_txt}.\n"
        f"Sample-efficiency note: QAE used ~{speedup:.0f}x fewer samples at the "
        "benchmark precision (simulator run — no wall-clock claim).\n\n"
        "Write 2-3 sentences of plain-English risk commentary for the desk head: "
        "what the number means under this scenario and what to watch today. "
        "All data is synthetic; do not add disclaimers, headers, or bullet points."
    )
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=REGION)
        resp = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            # Opus 5 spends tokens on a reasoning block before the text
            # answer, so the budget must cover both
            inferenceConfig={"maxTokens": 2000},
        )
        blocks = resp["output"]["message"]["content"]
        text = next(b["text"] for b in blocks if "text" in b).strip()
        return text, "bedrock"
    except Exception:
        return (
            _OFFLINE_TEMPLATE.format(
                scenario=scenario_name, pair=pair, var_pct=var_estimate,
                var_usd=var_usd, agree_txt=agree_txt, speedup=speedup,
            ),
            "offline",
        )
