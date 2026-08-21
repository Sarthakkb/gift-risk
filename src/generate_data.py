"""Generate synthetic position datasets (returns + metadata) to data/ CSVs.

Run:  python -m src.generate_data
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.market_data import PAIR_PARAMS, generate_returns, price_path, summary_stats
from src.metadata import make_position_metadata

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_position(pair: str, seed: int) -> tuple[pd.DataFrame, dict]:
    returns = generate_returns(pair, n_days=1500, seed=seed)
    rates = price_path(pair, returns)
    dates = pd.bdate_range(end="2026-08-21", periods=len(returns))
    df = pd.DataFrame({"date": dates, "fx_rate": rates, "log_return": returns})
    meta = make_position_metadata(pair)
    return df, meta


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    all_meta = {}
    for i, pair in enumerate(PAIR_PARAMS):
        df, meta = build_position(pair, seed=42 + i)
        slug = pair.replace("/", "_").lower()
        csv_path = DATA_DIR / f"{slug}.csv"
        df.to_csv(csv_path, index=False)
        all_meta[pair] = meta

        s = summary_stats(df["log_return"].to_numpy())
        print(f"\n{pair}  ->  {csv_path.name}  ({len(df)} rows)")
        print(f"  entity: {meta['entity_name']}  |  desk: {meta['desk_name']}")
        print(
            f"  ann_vol={s['ann_vol']:.2%}  skew={s['skewness']:+.2f}  "
            f"ex_kurt={s['excess_kurtosis']:.2f}  min={s['min']:+.4f}  max={s['max']:+.4f}"
        )

    meta_path = DATA_DIR / "positions_metadata.json"
    meta_path.write_text(json.dumps(all_meta, indent=2))
    print(f"\nmetadata -> {meta_path.name}")


if __name__ == "__main__":
    main()
