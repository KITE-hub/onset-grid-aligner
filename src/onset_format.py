from __future__ import annotations


def signed(value: float) -> str:
    return f"{round(float(value), 2) + 0.0:+.2f}"
