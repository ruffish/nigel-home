from __future__ import annotations

import json
from typing import Any


class UserVisibleError(Exception):
    pass


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception as err:  # noqa: BLE001
        raise UserVisibleError(f"Model returned invalid JSON: {err}") from err


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))
