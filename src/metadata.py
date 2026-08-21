"""Faker-based synthetic position metadata for GIFT Risk.

Metadata generation is deliberately separate from the statistical market-data
engine (src/market_data.py): distributions drive the risk math; metadata makes
positions legible to a treasury officer. All entities, IDs and counterparties
are SYNTHETIC.
"""

from __future__ import annotations

from faker import Faker

fake = Faker("en_IN")
Faker.seed(7)

# Realistic-sounding GIFT IFSC entity/desk building blocks
_ENTITY_SUFFIXES = ["IBU", "IFSC Banking Unit", "(IFSC Branch)"]
_BANKS = [
    "Axis Bank", "ICICI Bank", "HDFC Bank", "Kotak Mahindra Bank",
    "State Bank of India", "Standard Chartered", "Barclays", "MUFG",
    "DBS Bank", "Emirates NBD", "First Abu Dhabi Bank", "JP Morgan",
]
_DESKS = {
    "USD/INR": "Cross-Border Treasury Desk — USD",
    "SGD/INR": "Asia FX & Remittances Desk",
    "AED/INR": "Gulf Corridor Trade Finance Desk",
}

# Hedge book parameters: hand-set (like scenarios.py), not Faker-random —
# these need to be domain-coherent, not arbitrary. "exposure_type" sets
# which way an incremental hedge trades (see src/hedge.py);
# "current_hedge_ratio" is the fraction of notional already covered by
# existing forwards, feeding the Hedge Ratio tab's starting point.
_EXPOSURE = {
    "USD/INR": {
        "exposure_type": "payable",
        "exposure_note": "Offshore USD loan drawdown — desk owes USD at maturity",
        "current_hedge_ratio": 0.35,
    },
    "SGD/INR": {
        "exposure_type": "receivable",
        "exposure_note": "SGD trade & remittance receivables — desk is owed SGD",
        "current_hedge_ratio": 0.55,
    },
    "AED/INR": {
        "exposure_type": "payable",
        "exposure_note": "LC-backed import payables to Gulf suppliers — desk owes AED",
        "current_hedge_ratio": 0.20,
    },
}


def make_position_metadata(pair: str, valuation_date: str = "2026-08-21") -> dict:
    """Generate one synthetic GIFT IFSC position record for a currency pair."""
    bank = fake.random_element(_BANKS)
    counterparty = fake.random_element([b for b in _BANKS if b != bank])
    exposure = _EXPOSURE.get(pair, {"exposure_type": "payable", "exposure_note": "", "current_hedge_ratio": 0.5})
    return {
        "position_id": f"GIFT-{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}-{fake.random_number(digits=6, fix_len=True)}",
        "entity_name": f"{bank} {fake.random_element(_ENTITY_SUFFIXES)}, GIFT City",
        "desk_name": _DESKS.get(pair, "Treasury Desk"),
        "currency_pair": pair,
        "valuation_date": valuation_date,
        "counterparty": f"{counterparty} {fake.random_element(_ENTITY_SUFFIXES)}",
        "trader_ref": fake.name(),
        **exposure,
    }
