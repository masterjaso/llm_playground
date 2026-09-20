"""Operator CLI: flashmini-data (v4). Idempotent + resumable commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cache as cache_mod
from . import hf_store, manifests
from . import recipes as recipes_mod
from . import registry as registry_mod
from . import source as source_mod


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def cmd_auth_check(_args) -> int:
    try:
        info = hf_store.whoami()
    except RuntimeError as exc:
        print(f"auth: MISSING ({exc})")
        return 2
    print(f"auth: OK user={info.get('name','?')}")
    return 0


def cmd_source_list(args) -> int:
    reg = registry_mod.load_registry(Path(args.registry))
    for sid, src in reg["sources"].items():
        print(f"{sid} domain={src.get('domain')} class={src.get('redistribution_class')} "
              f"dataset={src.get('dataset_id')}")
    return 0


def cmd_source_probe(args) -> int:
    reg = registry_mod.load_registry(Path(args.registry))
    names = args.sources or list(reg["sources"])
    if args.all:
        names = list(reg["sources"])
    failed = 0
    for sid in names:
        src = dict(reg["sources"][sid])
        src["source_id"] = sid
        res = source_mod.stream_source_window(src, limit=args.limit)
        print(f"{sid}: status={res.status} reason={res.reason} "
              f"records={len(res.records)}")
        if res.status != "OK":
            failed += 1
    return 1 if failed else 0


def cmd_source_lock(args) -> int:
    from huggingface_hub import HfApi
    reg = registry_mod.load_registry(Path(args.registry))
    lock_path = Path(args.out)
    existing = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    token = hf_store.load_token()
    api = HfApi(token=token)
    for sid, src in reg["sources"].items():
        dsid = src["dataset_id"]
        try:
            info = api.dataset_info(dsid)
            sha = getattr(info, "sha", "") or ""
            card = ""
            try:
                card_file = api.hf_hub_download(dsid, "README.md", repo_type="dataset",
                                                revision=sha or None)
                import hashlib as _hl
                card = _hl.sha256(Path(card_file).read_bytes()).hexdigest()
            except Exception:
                card = ""
            existing[sid] = {
                "dataset_id": dsid,
                "revision": sha,
                "config": src.get("config"),
                "split": src.get("split", "train"),
                "license": getattr(info, "license", None) or src.get("license", ""),
                "gated": bool(getattr(info, "gated", False)),
                "redistribution_class": src.get("redistribution_class"),
                "card_sha256": card,
            }
            print(f"locked {sid} rev={(sha[:12] if sha else '?')}")
        except Exception as exc:
            print(f"lock FAILED {sid}: {type(exc).__name__}: {str(exc)[:160]}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(f"wrote {lock_path} ({len(existing)} sources)")
    return 0


def cmd_plan(args) -> int:
    recipe = recipes_mod.load_recipe(Path(args.recipe))
    targets = recipes_mod.domain_token_targets(recipe)
    print(f"recipe={recipe['name']} target={recipe['target_tokens']} seed={recipe['seed']}")
    for domain, tokens in targets.items():
        print(f"  {domain}: {tokens} tokens "
              f"(sources={recipe['domains'][domain]['sources']})")
    print(f"recipe_hash={recipes_mod.recipe_hash(recipe)}")
    return 0


def cmd_status(args) -> int:
    state = manifests.load_state(Path(args.state))
    root = cache_mod.cache_root(args.cache_dir)
    st = cache_mod.stats(root, cache_mod.cache_max_bytes(
        int(args.cache_gb * 1024 ** 3) if args.cache_gb else None))
    from .status import render_dashboard
    print(render_dashboard(state, cache_used_bytes=st.used_bytes,
                           hf_revision=state.get("hf_revision", "")))
    return 0


def cmd_cache_status(args) -> int:
    root = cache_mod.cache_root(args.cache_dir)
    st = cache_mod.stats(root, cache_mod.cache_max_bytes(
        int(args.cache_gb * 1024 ** 3) if args.cache_gb else None))
    print(f"root={st.root} used={st.used_bytes / 1024**3:.2f}GiB "
          f"max={st.max_bytes / 1024**3:.1f}GiB files={st.files} "
          f"disk_avail={st.avail_bytes / 1024**3:.1f}GiB")
    return 0


def cmd_cache_prune(args) -> int:
    root = cache_mod.cache_root(args.cache_dir)
    evicted = cache_mod.enforce_bound(root, cache_mod.cache_max_bytes(
        int(args.cache_gb * 1024 ** 3) if args.cache_gb else None))
    print(f"evicted {len(evicted)} files")
    for p in evicted[:20]:
        print(f"  {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flashmini-data")
    sub = p.add_subparsers(dest="cmd", required=True)
    dflt_reg = str(_repo_root() / "training_data/registry/sources.yaml")
    dflt_state = str(_repo_root() / "training_data/manifests/build_state.json")

    c = sub.add_parser("auth-check"); c.set_defaults(func=cmd_auth_check)
    c = sub.add_parser("source-list")
    c.add_argument("--registry", default=dflt_reg); c.set_defaults(func=cmd_source_list)
    c = sub.add_parser("source-probe")
    c.add_argument("--registry", default=dflt_reg)
    c.add_argument("--all", action="store_true")
    c.add_argument("--sources", nargs="*", default=None)
    c.add_argument("--limit", type=int, default=5)
    c.set_defaults(func=cmd_source_probe)
    c = sub.add_parser("source-lock")
    c.add_argument("--registry", default=dflt_reg)
    c.add_argument("--out", default=str(_repo_root() / "training_data/registry/source_snapshot.lock.json"))
    c.set_defaults(func=cmd_source_lock)
    c = sub.add_parser("plan")
    c.add_argument("--recipe", required=True); c.set_defaults(func=cmd_plan)
    _register_build_commands(sub, dflt_reg, dflt_state)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def _register_build_commands(sub, dflt_reg: str, dflt_state: str) -> None:
    from . import build as build_mod
    for name in ("build", "resume"):
        c = sub.add_parser(name)
        c.add_argument("--recipe", required=True)
        c.add_argument("--registry", default=dflt_reg)
        c.add_argument("--state", default=dflt_state)
        c.add_argument("--shard-docs", type=int, default=2000)
        c.add_argument("--max-docs", type=int, default=20000)
        c.add_argument("--per-source-docs", type=int, default=None,
                       help="Cap per source window so no single source consumes the budget")
        c.add_argument("--window", type=int, default=500)
        c.add_argument("--out-dir", default=str(_repo_root() / "training_data/manifests/shards"))
        c.add_argument("--cache-dir", default=None)
        c.add_argument("--cache-gb", type=float, default=None)
        c.add_argument("--split-salt", default="flashmini-v4-split-v1")
        c.add_argument("--hf-prefix", default="shards",
                       help="Remote folder for this release (avoids collisions)")
        c.add_argument("--no-publish", action="store_true")
        c.set_defaults(func=build_mod.cmd_build)
    c = sub.add_parser("publish")
    c.add_argument("--state", default=dflt_state)
    c.add_argument("--shard-dir", default=str(_repo_root() / "training_data/manifests/shards"))
    c.set_defaults(func=build_mod.cmd_publish)
    c = sub.add_parser("verify")
    c.add_argument("--recipe", required=True)
    c.add_argument("--manifest", default=str(_repo_root() / "training_data/manifests/corpus_manifest.json"))
    c.set_defaults(func=build_mod.cmd_verify)
    c = sub.add_parser("freeze")
    c.add_argument("--recipe", required=True)
    c.add_argument("--state", default=dflt_state)
    c.add_argument("--registry", default=dflt_reg)
    c.add_argument("--split-salt", default="flashmini-v4-split-v1")
    c.add_argument("--tokenizer", default="gpt2")
    c.add_argument("--manifest", default=str(_repo_root() / "training_data/manifests/corpus_manifest.json"))
    c.set_defaults(func=build_mod.cmd_freeze)
    for name, fn in (("train-smoke", build_mod.cmd_train_smoke),
                     ("resume-check", build_mod.cmd_resume_check)):
        c = sub.add_parser(name)
        c.add_argument("--manifest", default=str(_repo_root() / "training_data/manifests/corpus_manifest.json"))
        c.add_argument("--split", default="train")
        c.add_argument("--seq-len", type=int, default=2048)
        c.add_argument("--batch-size", type=int, default=4)
        c.add_argument("--batches", type=int, default=10)
        c.add_argument("--seed", type=int, default=0)
        c.add_argument("--epoch", type=int, default=0)
        c.add_argument("--consumed-batches", type=int, default=0)
        c.add_argument("--tokenizer", default="gpt2")
        c.add_argument("--hf-repo", default=None)
        c.add_argument("--revision", default=None)
        c.add_argument("--cache-dir", default=None)
        c.add_argument("--cache-gb", type=float, default=None)
        c.add_argument("--local-base", default=None,
                       help="Read shards from a local directory instead of the Hub")
        c.add_argument("--max-open-shards", type=int, default=4)
        c.set_defaults(func=fn)
    c = sub.add_parser("status")
    c.add_argument("--state", default=dflt_state)
    c.add_argument("--cache-dir", default=None)
    c.add_argument("--cache-gb", type=float, default=None)
    c.set_defaults(func=cmd_status)
    c = sub.add_parser("cache-status")
    c.add_argument("--cache-dir", default=None)
    c.add_argument("--cache-gb", type=float, default=None)
    c.set_defaults(func=cmd_cache_status)
    c = sub.add_parser("cache-prune")
    c.add_argument("--cache-dir", default=None)
    c.add_argument("--cache-gb", type=float, default=None)
    c.set_defaults(func=cmd_cache_prune)



if __name__ == "__main__":
    sys.exit(main())
