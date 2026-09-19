"""Declarative recipe loading + hashing (v4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

RECIPE_VERSION = "flashmini-recipe-v1"


def load_recipe(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("recipe must be a mapping")
    for field in ("name", "target_tokens", "seed", "domains"):
        if field not in raw:
            raise ValueError(f"recipe missing required field '{field}'")
    if not isinstance(raw["domains"], dict) or not raw["domains"]:
        raise ValueError("recipe 'domains' must be a non-empty mapping")
    total_w = sum(float(d.get("weight", 0)) for d in raw["domains"].values())
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"recipe domain weights must sum to 1.0 (got {total_w})")
    for dname, dom in raw["domains"].items():
        if "sources" not in dom or not dom["sources"]:
            raise ValueError(f"recipe domain {dname}: missing source list")
    return raw


def recipe_hash(recipe: dict) -> str:
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode("utf-8")).hexdigest()


def domain_token_targets(recipe: dict) -> dict[str, int]:
    total = int(recipe["target_tokens"])
    return {name: int(total * float(dom["weight"]))
            for name, dom in recipe["domains"].items()}
