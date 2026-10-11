"""Single source of truth for model tiers and list prices.

Imported by every harness (react_api, crmarena), the spend ledger and the
cost estimator, so a repricing is a one-line change here.

Prices are USD per million tokens (input, output), Anthropic first-party
list prices as of 2026-09.
"""
from __future__ import annotations

TIERS = ["cheap", "mid", "frontier"]
MODELS = {"cheap": "claude-haiku-4-5", "mid": "claude-sonnet-5-5", "frontier": "claude-opus-5-5"}
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
CACHE_WRITE_MULT = 1.25   # 5-minute cache write, relative to the input price
CACHE_READ_MULT = 0.10


def bare_model(model: str) -> str:
    """'anthropic/claude-haiku-4-5' -> 'claude-haiku-4-5'."""
    return model.split("/", 1)[1] if "/" in model else model


def model_for(tier_or_model: str) -> str:
    return MODELS.get(tier_or_model, tier_or_model)


def price_of(model: str, prices: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
    table = PRICES_PER_MTOK if prices is None else prices
    m = bare_model(model_for(model))
    if m not in table:
        raise KeyError(f"no list price for model {model!r}; add it to pace/prices.py")
    return table[m]


def repriced(factors: dict[str, float]) -> dict[str, tuple[float, float]]:
    """The price table after a repricing event: {tier or model: factor} scales both the
    input and the output price of that model, e.g. {"frontier": 1.5}."""
    out = dict(PRICES_PER_MTOK)
    for k, f in factors.items():
        m = bare_model(model_for(k))
        pin, pout = out[m]
        out[m] = (pin * float(f), pout * float(f))
    return out


def call_cost(model: str, in_tokens: int, out_tokens: int,
              cache_write_tokens: int = 0, cache_read_tokens: int = 0,
              prices: dict[str, tuple[float, float]] | None = None) -> float:
    """USD for one call. `in_tokens` excludes cache reads/writes (Anthropic usage semantics).
    `prices` overrides the list-price table (see `repriced`)."""
    pin, pout = price_of(model, prices)
    return (in_tokens * pin + out_tokens * pout
            + cache_write_tokens * pin * CACHE_WRITE_MULT
            + cache_read_tokens * pin * CACHE_READ_MULT) / 1e6
