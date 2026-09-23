"""Declarative recipes, stage aggregation, and exact token contracts.

Recipes are the durable curriculum contract.  The ``full`` recipes are derived
from their stage files and are rejected when a hand-edited aggregate drifts.
This keeps a production target expressed in exact train tokens rather than in
document counts or approximate percentages.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import yaml

RECIPE_VERSION = "flashmini-recipe-v1"


def _fraction(value: object) -> Fraction:
    """Parse a YAML weight without introducing binary-float rounding."""
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def _integer_targets(total: int, weights: dict[str, Fraction]) -> dict[str, int]:
    """Allocate ``total`` exactly using largest-remainder rounding."""
    raw = {name: _fraction(total) * weight for name, weight in weights.items()}
    base = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(base.values())
    ranked = sorted(raw, key=lambda name: (raw[name] - base[name], name), reverse=True)
    for name in ranked[:remainder]:
        base[name] += 1
    return base


def load_recipe(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError("recipe must be a mapping")
    for field in ("name", "target_tokens", "seed", "domains"):
        if field not in raw:
            raise ValueError(f"recipe missing required field '{field}'")
    if not isinstance(raw["domains"], dict) or not raw["domains"]:
        raise ValueError("recipe 'domains' must be a non-empty mapping")
    total_w = sum((_fraction(d.get("weight", 0)) for d in raw["domains"].values()),
                  Fraction(0))
    if total_w != 1:
        raise ValueError(f"recipe domain weights must sum to 1.0 (got {float(total_w)})")
    if int(raw["target_tokens"]) != raw["target_tokens"] or int(raw["target_tokens"]) <= 0:
        raise ValueError("recipe target_tokens must be a positive integer")
    for dname, dom in raw["domains"].items():
        if "sources" not in dom or not dom["sources"]:
            raise ValueError(f"recipe domain {dname}: missing source list")
        if _fraction(dom.get("weight", 0)) <= 0:
            raise ValueError(f"recipe domain {dname}: weight must be positive")
    # A full recipe carries stage names.  Validate the aggregate at load time
    # when its sibling stage files are available; callers can opt out for
    # isolated schema probes by omitting the ``stages`` field.
    stages = raw.get("stages")
    if stages:
        stage_recipes = []
        for stage in stages:
            stage_path = Path(path).parent / f"{stage}.yaml"
            if not stage_path.exists():
                raise ValueError(f"recipe stage not found: {stage_path}")
            stage_recipes.append(load_recipe(stage_path))
        validate_stage_aggregate(raw, stage_recipes)
    return raw


def recipe_hash(recipe: dict) -> str:
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode("utf-8")).hexdigest()


def domain_token_targets(recipe: dict) -> dict[str, int]:
    total = int(recipe["target_tokens"])
    weights = {name: _fraction(dom["weight"]) for name, dom in recipe["domains"].items()}
    return _integer_targets(total, weights)


def aggregate_stage_recipes(stage_recipes: list[dict]) -> dict:
    """Return the exact weighted aggregate of stage contracts.

    The result includes integer domain targets, normalized weights, and the
    union of stage sources.  Stage order is significant for token totals but
    not for the deterministic output representation.
    """
    if not stage_recipes:
        raise ValueError("at least one stage recipe is required")
    total = sum(int(recipe["target_tokens"]) for recipe in stage_recipes)
    raw: dict[str, Fraction] = {}
    sources: dict[str, set[str]] = {}
    for recipe in stage_recipes:
        stage_total = int(recipe["target_tokens"])
        for name, domain in recipe["domains"].items():
            raw[name] = raw.get(name, Fraction(0)) + _fraction(domain["weight"]) * stage_total
            sources.setdefault(name, set()).update(str(s) for s in domain.get("sources", []))
    targets = _integer_targets(total, {name: value / total for name, value in raw.items()})
    domains = {
        name: {
            "weight": float(Fraction(target, total)),
            "target_tokens": target,
            "sources": sorted(sources.get(name, set())),
        }
        for name, target in sorted(targets.items())
    }
    return {
        "target_tokens": total,
        "domain_targets": targets,
        "domains": domains,
        "stage_names": [str(recipe.get("name", "")) for recipe in stage_recipes],
    }


def validate_stage_aggregate(full_recipe: dict, stage_recipes: list[dict]) -> dict:
    """Raise with a useful diff if a full recipe drifts from its stages."""
    expected = aggregate_stage_recipes(stage_recipes)
    actual = domain_token_targets(full_recipe)
    if int(full_recipe["target_tokens"]) != expected["target_tokens"]:
        raise ValueError(
            f"recipe {full_recipe.get('name', '')}: target_tokens drift: "
            f"{full_recipe['target_tokens']} != {expected['target_tokens']}")
    if actual != expected["domain_targets"]:
        diffs = {
            name: (actual.get(name, 0), expected["domain_targets"].get(name, 0))
            for name in sorted(set(actual) | set(expected["domain_targets"]))
            if actual.get(name, 0) != expected["domain_targets"].get(name, 0)
        }
        raise ValueError(f"recipe {full_recipe.get('name', '')}: stage aggregate drift: {diffs}")
    return expected
