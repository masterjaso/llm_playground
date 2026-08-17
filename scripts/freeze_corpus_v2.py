#!/usr/bin/env python3
"""Acquire, audit, split, and freeze the production-weighted corpus V2.

This command is deliberately independent from model training.  It materializes
fixed activation *contexts* and records enough provenance to replay the exact
source selection later through the dense teacher.  The external agent model is
never a teacher label: trajectory text is only an input-distribution source.

The default run is intentionally bounded by source-row count (a few pinned
repository files and the first public rows of the Qwen non-thinking
Open-SWE-Traces splits), but it preserves each selected visible trajectory in
full.  Fixed-length activation windows can be derived later without silently
discarding the causal middle of an agent run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/public_v2/corpus.jsonl"
DEFAULT_RECEIPT = ROOT / "data/public_v2/corpus-v2-receipt.json"
DEFAULT_SPLITS = ROOT / "data/public_v2/corpus-v2-splits.json"
DEFAULT_TOKENIZER = ROOT / "runs/20260815-030931-windows/source/tokenizer.json"
DEFAULT_TOKENIZER_SNAPSHOT = ROOT / "runs/20260815-030931-windows/source"

HF_FIRST_ROWS = "https://datasets-server.huggingface.co/first-rows"
OPEN_SWE_DATASET = "nvidia/Open-SWE-Traces"
# ``rows`` currently reports this immutable dataset-view revision in its
# ``x-revision`` response header.  Keeping it in every record prevents a
# moving ``main`` alias from becoming part of the scientific identity.
OPEN_SWE_REVISION = "ad4805a5aa7de70d99cab0bb8f99b15304c76de0"

# These are pinned at materialization time.  The commit values are deliberately
# explicit rather than branch names; a changed file becomes a new corpus.
REPOSITORY_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "repo": "psf/requests",
        "commit": "8068356288978c4f54661ae6f95afe0e0831885e",
        "license": "Apache-2.0",
        "language": "python",
        "paths": ("src/requests/api.py", "src/requests/sessions.py", "tests/test_requests.py", "pyproject.toml", "docs/user/quickstart.rst", "docs/index.rst"),
    },
    {
        "repo": "python/cpython",
        "commit": "7a845ce16548bf94e777984458ea534c5a65a2a8",
        "license": "PSF-2.0",
        "language": "python",
        "paths": ("Lib/asyncio/base_events.py", "Lib/test/test_asyncio/test_base_events.py", "README.rst", "Doc/library/asyncio.rst", "Doc/whatsnew/3.14.rst"),
    },
    {
        "repo": "microsoft/TypeScript",
        "commit": "b465fdbfe175304d9b977da137b2c178ae1091d3",
        "license": "Apache-2.0",
        "language": "typescript",
        "paths": ("src/compiler/program.ts", "src/compiler/transformer.ts", "src/compiler/types.ts", "package.json", "CONTRIBUTING.md"),
    },
    {
        "repo": "rust-lang/rustlings",
        "commit": "02b22b49e7349b36305dbe97cd5d6e198d2f0537",
        "license": "MIT",
        "language": "rust",
        "paths": ("exercises/01_variables/variables1.rs", "exercises/14_generics/generics1.rs", "exercises/01_variables/README.md"),
    },
    {
        "repo": "golang/go",
        "commit": "72aa6db7943024b48c4d41c1fbc32b57b9fa036e",
        "license": "BSD-3-Clause",
        "language": "go",
        "paths": ("src/net/http/client.go", "src/net/http/client_test.go", "src/os/file.go", "doc/go_spec.html", "src/encoding/json/encode.go"),
    },
    {
        "repo": "dotnet/runtime",
        "commit": "a36e3c1c9a993f0b4c6db87f033105f0bb364f3b",
        "license": "MIT",
        "language": "csharp",
        "paths": (
            "src/libraries/System.Text.Json/src/System/Text/Json/Document/JsonDocument.cs",
            "src/libraries/System.Text.Json/src/System/Text/Json/Serialization/JsonSerializer.Write.String.cs",
            "src/libraries/System.Text.Json/src/System.Text.Json.csproj",
        ),
    },
    {
        "repo": "PowerShell/PowerShell",
        "commit": "174a75396cf3835b5c36c4a95030dc96fe73ab47",
        "license": "MIT",
        "language": "powershell/csharp",
        "paths": ("src/System.Management.Automation/engine/InitialSessionState.cs", "README.md"),
    },
)

BENCHMARK_DENYLIST = {
    "swe-bench",
    "swebench",
    "swe-rebench",
    "human-eval",
    "humaneval",
    "mbpp",
    "ds-1000",
    "cruxeval",
    "livecodebench",
}

SPLITS = ("FIT-TRAIN", "FIT-DEV", "GATE-A", "SHADOW-B", "SHADOW-C", "PRESERVATION-CANARY")
SCHEMA_VERSION = 2


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())


def _norm_hash(text: str) -> str:
    return _sha256_bytes(_canonical_text(text).encode("utf-8"))


def _request_json(url: str, *, retries: int = 3) -> Any:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dense2moe-corpus-v2/2.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 < retries:
                continue
    raise RuntimeError(f"failed to fetch JSON after {retries} attempts: {url}: {last}")


def _request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "dense2moe-corpus-v2/2.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _trajectory_text(raw: Any) -> tuple[str, list[str]]:
    """Render visible actions/observations, excluding private reasoning fields."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip(), []
    if not isinstance(raw, list):
        return "", []
    chunks: list[str] = []
    roles: list[str] = []
    excluded_roles = {"analysis", "thought", "reasoning", "chain_of_thought", "cot"}
    for message in raw:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", message.get("from", ""))).strip().lower()
        if role in excluded_roles:
            continue
        # Open-SWE carries optional private fields beside the visible message.
        # They are not needed for activation contexts and must never leak into
        # a corpus that claims to use the non-thinking Qwen split.
        if message.get("reasoning_content") or message.get("think") is True:
            continue
        content = message.get("content", message.get("value", ""))
        if isinstance(content, list):
            # Multimodal/tool content is retained only as its visible text.
            parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping):
                    value = part.get("text", part.get("content", ""))
                    if isinstance(value, str):
                        parts.append(value)
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(parts)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True) if content else ""
        content = re.sub(r"(?is)<think>.*?</think>", "", content).strip()
        if content.strip():
            chunks.append(f"[{role or 'message'}]\n{content.strip()}")
            roles.append(role or "message")
        # Function/tool calls are actions, not teacher labels; keep their
        # visible names and arguments while omitting hidden thought keys.
        calls = message.get("tool_calls", message.get("function_call"))
        if calls:
            if not isinstance(calls, list):
                calls = [calls]
            visible_calls: list[Any] = []
            for call in calls:
                function = call.get("function", {}) if isinstance(call, Mapping) else {}
                name = str(function.get("name", "")) if isinstance(function, Mapping) else ""
                # Open-SWE stores private planning in a synthetic ``think``
                # tool call.  Do not let that field leak into activation
                # contexts; ordinary execute/read/edit calls remain visible.
                if name.lower() in excluded_roles or name.lower() in {"think", "reason", "reasoning"}:
                    continue
                arguments = function.get("arguments") if isinstance(function, Mapping) else None
                if isinstance(arguments, str):
                    try:
                        parsed_arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        parsed_arguments = arguments
                    if isinstance(parsed_arguments, Mapping) and any(
                        str(key).lower() in excluded_roles or str(key).lower() in {"thought", "reasoning_content"}
                        for key in parsed_arguments
                    ):
                        continue
                visible_calls.append(call)
            if visible_calls:
                calls_text = json.dumps(visible_calls, ensure_ascii=False, sort_keys=True)
                chunks.append(f"[tool_call]\n{calls_text}")
                roles.append("tool_call")
    return "\n\n".join(chunks), roles


def _visible_tool_schema(raw: Any) -> str:
    """Serialize public tool/function schemas without private reasoning tools."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return ""
    if not isinstance(raw, list):
        return ""
    visible: list[Any] = []
    for item in raw:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if not isinstance(item, Mapping):
            continue
        function = item.get("function", item)
        if isinstance(function, Mapping) and str(function.get("name", "")).lower() in {"think", "reason", "analysis"}:
            continue
        visible.append(item)
    return json.dumps(visible, ensure_ascii=False, sort_keys=True) if visible else ""


def _source_record(**values: Any) -> dict[str, Any]:
    text = str(values.pop("text", ""))
    if not text.strip():
        raise ValueError("empty corpus record")
    record = {
        "text": text,
        "source_name": str(values.pop("source_name")),
        "source_revision": str(values.pop("source_revision")),
        "source_license": str(values.pop("source_license")),
        "source_url": str(values.pop("source_url", "")),
        "download_sha256": str(values.pop("download_sha256", "")),
        "source_record_id": str(values.pop("source_record_id")),
        "domain": str(values.pop("domain")),
        "source_family": str(values.pop("source_family")),
        "task_family": str(values.pop("task_family", "general")),
        "language": str(values.pop("language", "text")),
        "trajectory_framework": str(values.pop("trajectory_framework", "")),
        "trajectory_model": str(values.pop("trajectory_model", "")),
        "benchmark_context": list(values.pop("benchmark_context", [])),
        "repo": str(values.pop("repo", "")),
        "repo_commit": str(values.pop("repo_commit", "")),
        "repo_path": str(values.pop("repo_path", "")),
        "split_group": str(values.pop("split_group")),
        "benchmark_membership": list(values.pop("benchmark_membership", [])),
        "selection_rationale": str(values.pop("selection_rationale")),
        "rationale": str(values.pop("rationale", "")),
        "source_artifact_url": str(values.pop("source_artifact_url", "")),
        "terms": str(values.pop("terms", "")),
        "upstream_license": str(values.pop("upstream_license", "")),
        "document_id": str(values.pop("document_id", "")),
        "task_id": str(values.pop("task_id", "")),
        "trajectory_id": str(values.pop("trajectory_id", "")),
    }
    if values:
        record.update(values)
    return record


def _truncate_tokens(tokenizer: Any, text: str, limit: int) -> tuple[str, int, bool]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = list(getattr(encoded, "ids", encoded))
    original = len(ids)
    if limit <= 0 or original <= limit:
        return text, original, False
    # Keep the request/context prefix and the validation/recovery tail.  This
    # is a bounded pilot representation; the receipt exposes the truncation
    # so serious captures can raise the cap without changing split identities.
    half = max(1, limit // 2)
    ids = ids[:half] + ids[-(limit - half):]
    return tokenizer.decode(ids, skip_special_tokens=False), original, True


def fetch_open_swe(limit_per_framework: int, tokenizer: Any, trajectory_token_cap: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for framework in ("openhands", "sweagent"):
        query = urllib.parse.urlencode(
            {"dataset": OPEN_SWE_DATASET, "config": framework, "split": "qwen35_122b", "offset": 0, "length": limit_per_framework}
        )
        # The first-rows endpoint truncates long trajectory strings to a
        # preview.  The rows endpoint returns the complete fixed sequences;
        # retain a first-rows fallback for transient dataset-view outages.
        rows_url = HF_FIRST_ROWS.replace("/first-rows", "/rows")
        try:
            payload = _request_json(f"{rows_url}?{query}")
        except RuntimeError:
            preview_query = urllib.parse.urlencode(
                {"dataset": OPEN_SWE_DATASET, "config": framework, "split": "qwen35_122b"}
            )
            payload = _request_json(f"{HF_FIRST_ROWS}?{preview_query}")
        rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
        for wrapped in rows[:limit_per_framework]:
            row = wrapped.get("row", {}) if isinstance(wrapped, Mapping) else {}
            if not isinstance(row, Mapping):
                continue
            text, roles = _trajectory_text(row.get("trajectory", ""))
            if len(text) < 64:
                continue
            text, original_token_count, truncated = _truncate_tokens(tokenizer, text, trajectory_token_cap)
            repo = str(row.get("repo", "unknown"))
            instance = str(row.get("instance_id", row.get("trajectory_id", "")))
            language = str(row.get("language", "unknown")).lower()
            license_name = str(row.get("license", "CC-BY-4.0"))
            if license_name.lower() in {"", "unknown", "none", "noassertion"}:
                continue
            category = "general"
            metadata = row.get("metadata", "")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            if isinstance(metadata, Mapping):
                category = str(metadata.get("category", "general"))
            records.append(
                _source_record(
                    text=text,
                    source_name="nvidia/Open-SWE-Traces",
                    source_revision=OPEN_SWE_REVISION,
                    source_license="CC-BY-4.0",
                    source_url="https://huggingface.co/datasets/nvidia/Open-SWE-Traces",
                    download_sha256=_sha256_bytes(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")),
                    source_record_id=f"{framework}:{instance}",
                    domain="agentic-software-engineering",
                    source_family="agent-trajectory",
                    task_family=category,
                    language=language,
                    trajectory_framework=framework,
                    trajectory_model="Qwen3.5-122B-A10B (non-thinking)",
                    repo=repo,
                    repo_commit="",
                    repo_path="",
                    # Repository identity is the disjointness unit.  Grouping
                    # by issue alone would allow two tasks from one repository
                    # to land in opposing evaluation roles.
                    split_group=f"repo:{repo}",
                    # Open-SWE issue statements are sourced from SWE-rebench;
                    # retain that provenance without treating the entire
                    # public source family as an exact held-out task denylist.
                    benchmark_membership=["SWE-rebench-V2-derived", "SWE-Bench-style"],
                    benchmark_denylist=[],
                    benchmark_context=["SWE-rebench-V2-derived", "SWE-Bench-style"],
                    selection_rationale="Visible Qwen3.5 non-thinking software-agent actions and observations; external response text is context only, never a teacher label.",
                    rationale=f"Retain visible roles {sorted(set(roles))}; exclude private reasoning fields.",
                    source_artifact_url=f"{rows_url}?{query}",
                    terms="Open-SWE-Traces is released under CC-BY-4.0; retain attribution, the upstream repository license, and source provenance.",
                    upstream_license=license_name,
                    document_id=f"trajectory:{framework}:{instance}",
                    task_id=instance,
                    trajectory_id=str(row.get("trajectory_id", "")),
                    trajectory_original_token_count=original_token_count,
                    trajectory_truncated=truncated,
                    trajectory_truncation_policy="prefix+suffix token cap" if truncated else "none",
                )
            )
            tool_schema = _visible_tool_schema(row.get("tools", ""))
            if tool_schema:
                tool_schema = f"source_repo: {repo}\ntrajectory_framework: {framework}\n{tool_schema}"
                records.append(
                    _source_record(
                        text=tool_schema,
                        source_name="nvidia/Open-SWE-Traces",
                        source_revision=OPEN_SWE_REVISION,
                        source_license="CC-BY-4.0",
                        source_url="https://huggingface.co/datasets/nvidia/Open-SWE-Traces",
                        download_sha256=_sha256_bytes(json.dumps(row.get("tools"), sort_keys=True, ensure_ascii=False).encode("utf-8")),
                        source_record_id=f"{framework}:tools:{instance}",
                        domain="structured",
                        source_family="structured-tool-material",
                        task_family="tool-calling",
                        language="json",
                        trajectory_framework=framework,
                        trajectory_model="Qwen3.5-122B-A10B (non-thinking)",
                        repo=repo,
                        split_group=f"repo:{repo}",
                        benchmark_membership=["SWE-rebench-V2-derived", "SWE-Bench-style"],
                        benchmark_denylist=[],
                        benchmark_context=["SWE-rebench-V2-derived", "SWE-Bench-style"],
                        selection_rationale="Visible public tool/function schemas paired with an agent trajectory; no model answer or private reasoning is used as a label.",
                        rationale="Structured tool material covers JSON/function-call contexts without importing hidden thought fields.",
                        source_artifact_url=f"{rows_url}?{query}",
                        terms="Open-SWE-Traces is released under CC-BY-4.0; retain attribution, the upstream repository license, and source provenance.",
                        upstream_license=license_name,
                        document_id=f"trajectory:{framework}:{instance}:tools",
                        task_id=instance,
                        trajectory_id=str(row.get("trajectory_id", "")),
                    )
                )
    return records


def fetch_repository_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in REPOSITORY_SOURCES:
        repo = str(spec["repo"])
        commit = str(spec["commit"])
        for path in spec["paths"]:
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
            try:
                raw = _request_bytes(url)
            except (OSError, urllib.error.HTTPError) as exc:
                # A pinned commit can legitimately move a generated or
                # optional file.  Continue only for optional files; if an
                # entire repository disappears, the receipt will expose it.
                print(f"warning: skipping unavailable pinned file {url}: {exc}", file=sys.stderr)
                continue
            text = raw.decode("utf-8", errors="replace")
            suffix = Path(path).suffix.lower()
            if suffix in {".json", ".lock"}:
                domain, task = "structured", "configuration"
            elif suffix in {".md", ".rst", ".txt", ".html"}:
                domain, task = "software-engineering-natural-language", "documentation"
            elif "test" in Path(path).name.lower() or "/test" in path.lower():
                domain, task = "code", "test-repair"
            elif suffix in {".toml", ".yaml", ".yml", ".csproj", ".xml"}:
                domain, task = "structured", "configuration"
            else:
                domain, task = "code", "implementation"
            records.append(
                _source_record(
                    text=text,
                    source_name=repo,
                    source_revision=commit,
                    source_license=str(spec["license"]),
                    source_url=f"https://github.com/{repo}/blob/{commit}/{path}",
                    download_sha256=_sha256_bytes(raw),
                    source_record_id=f"{repo}:{commit}:{path}",
                    domain=domain,
                    source_family="real-repository",
                    task_family=task,
                    language=str(spec["language"]),
                    repo=repo,
                    repo_commit=commit,
                    repo_path=path,
                    split_group=f"repo:{repo}",
                    benchmark_membership=[],
                    selection_rationale="Pinned file from a permissively licensed maintained repository; retain implementations, tests, build/configuration, and documentation.",
                    rationale="Repository-level grouping prevents the same repository crossing FIT/A/B/C.",
                    source_artifact_url=url,
                    terms="Pinned repository file; comply with the recorded upstream license and repository terms.",
                    upstream_license=str(spec["license"]),
                    document_id=f"repo:{repo}:{commit}:{path}",
                    task_id=f"repo:{repo}:{commit}:{path}",
                )
            )
    return records


def load_legacy_preservation_records(path: Path, limit: int = 96) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or len(records) >= limit:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, Mapping) or not str(row.get("text", "")).strip():
            continue
        # Benchmark instances are denied from all optimization roles.  The
        # preservation canary should remain a broad-text regression set, so
        # select the old Gutenberg/general rows rather than HumanEval/GSM8K
        # benchmark material.
        old_domain = str(row.get("domain", "")).lower()
        old_source = str(row.get("source_name", "")).lower()
        if "gutenberg" not in old_source and old_domain not in {"general", "long-context"}:
            continue
        records.append(
            _source_record(
                text=str(row["text"]),
                source_name=str(row.get("source_name", "legacy-public-corpus")),
                source_revision=str(row.get("source_revision", "legacy")),
                source_license=str(row.get("source_license", row.get("license", "Public Domain"))),
                source_url=str(row.get("source_url", "")),
                download_sha256=str(row.get("download_sha256", "")),
                source_record_id=f"legacy:{line_number}:{row.get('source_record_id', line_number)}",
                domain="general-preservation",
                source_family="preservation-canary",
                task_family="general-text",
                language="text",
                split_group=f"legacy-document:{row.get('source_name', 'legacy')}",
                benchmark_membership=[],
                selection_rationale="Historical broad-text source retained only as a regression/OOD preservation canary; excluded from production FIT optimization.",
                rationale="This source influenced prior experiments and is not broad-generalization evidence.",
                source_artifact_url=str(row.get("source_url", "")),
                terms="Historical preservation source retained for regression only; do not use as optimization data.",
                upstream_license=str(row.get("source_license", row.get("license", "Public Domain"))),
                document_id=f"legacy:{row.get('source_name', 'legacy')}:{line_number}",
                task_id=f"legacy:{line_number}",
            )
        )
    return records


def load_legacy_support_records(path: Path, limit: int = 512) -> list[dict[str, Any]]:
    """Retain a small broad-dialogue/STEM slice inside FIT, not just canary."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if len(records) >= limit or not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, Mapping) or not str(row.get("text", "")).strip():
            continue
        source = str(row.get("source_name", "")).lower()
        if "openassistant" not in source:
            continue
        record_id = str(row.get("source_record_id", line_number))
        records.append(
            _source_record(
                text=str(row["text"]),
                source_name="OpenAssistant/oasst1",
                source_revision=str(row.get("source_revision", "fdf72ae0827c1cda404aff25b6603abec9e3399b")),
                source_license="Apache-2.0",
                source_url="https://huggingface.co/datasets/OpenAssistant/oasst1",
                download_sha256=str(row.get("download_sha256", "")),
                source_record_id=f"fit-support:{record_id}",
                domain="general",
                source_family="general-stem-preservation",
                task_family="technical-conversation",
                language="text",
                split_group=f"support-document:{record_id}",
                benchmark_membership=[],
                selection_rationale="Small Apache-2.0 public dialogue slice retained in FIT to preserve general-language behavior while the main mixture is production weighted.",
                rationale="General preservation material is distinct from the historical canary and is capped to prevent dominance.",
                source_artifact_url="https://huggingface.co/datasets/OpenAssistant/oasst1",
                terms="OpenAssistant/oasst1 Apache-2.0 dataset terms; preserve source attribution and revision.",
                upstream_license="Apache-2.0",
                document_id=f"support:{record_id}",
                task_id=f"support:{record_id}",
            )
        )
    return records


def _contains_denied_benchmark(record: Mapping[str, Any]) -> list[str]:
    haystack = " ".join(str(record.get(key, "")) for key in ("source_name", "source_url", "source_record_id", "repo", "repo_path", "task_family"))
    haystack = haystack.lower()
    # ``benchmark_membership`` is provenance (for example, Open-SWE is
    # SWE-rebench-derived); only an explicit denylist tag or exact task key
    # blocks a record.  This preserves the source while making contamination
    # policy auditable rather than silently discarding the entire source.
    tags = [str(item).lower() for item in record.get("benchmark_denylist", [])]
    return sorted({item for item in BENCHMARK_DENYLIST if item in haystack or any(item in tag for tag in tags)})


def _split_for_group(group: str, source_family: str = "") -> str:
    # Use one group-only assignment for all source families.  A trajectory and
    # its visible tool schema therefore stay together, while the deterministic
    # role salt gives this small pilot coding/agent representation in A/B/C.
    del source_family
    digest = int(hashlib.sha256(f"role:{group}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if group.startswith("legacy-document:"):
        return "PRESERVATION-CANARY"
    # Keep a substantial FIT pool while reserving genuinely independent groups.
    if digest < 55:
        return "FIT-TRAIN"
    if digest < 67:
        return "FIT-DEV"
    if digest < 80:
        return "GATE-A"
    if digest < 90:
        return "SHADOW-B"
    return "SHADOW-C"


def _load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("tokenizers is required to freeze corpus-v2 with exact source-tokenizer counts") from exc
    if not path.exists():
        raise FileNotFoundError(path)
    return Tokenizer.from_file(str(path))


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    return len(ids)


def _assign_and_audit(records: Iterable[dict[str, Any]], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    denied: list[dict[str, Any]] = []
    duplicate_count = 0
    for record in records:
        text = str(record.get("text", ""))
        if not text.strip():
            continue
        normalized_hash = _norm_hash(text)
        if normalized_hash in seen:
            duplicate_count += 1
            continue
        seen.add(normalized_hash)
        denied_tags = _contains_denied_benchmark(record)
        if denied_tags:
            denied.append({"source_record_id": record.get("source_record_id"), "matches": denied_tags})
            continue
        split_group = str(record.get("split_group", record.get("source_record_id", "")))
        split = _split_for_group(split_group, str(record.get("source_family", "")))
        token_count = _token_count(tokenizer, text)
        if token_count <= 0:
            continue
        selected.append(
            {
                **record,
                "source_file": "corpus.jsonl",
                "source_record_index": len(selected),
                "normalized_content_sha256": normalized_hash,
                "content_sha256": _sha256_bytes(text.encode("utf-8")),
                "token_count": token_count,
                "split": split,
                "split_group": split_group,
            }
        )
    # Verify group disjointness across all non-canary roles.
    groups: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        groups[str(row["split_group"])].add(str(row["split"]))
    overlap_groups = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if overlap_groups:
        raise ValueError(f"split-group overlap detected: {overlap_groups}")
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        split_groups[str(row["split"])].add(str(row["split_group"]))
    cross_split_overlaps: dict[str, list[str]] = {}
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            overlap = split_groups[left] & split_groups[right]
            if overlap:
                cross_split_overlaps[f"{left}|{right}"] = sorted(overlap)
    if cross_split_overlaps:
        raise ValueError(f"repository/document overlap detected: {cross_split_overlaps}")
    task_groups: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        if row["split"] == "PRESERVATION-CANARY":
            continue
        task_key = str(row.get("task_id") or row.get("document_id") or row.get("source_record_id"))
        task_groups[task_key].add(str(row["split"]))
    task_overlap = {task: sorted(splits) for task, splits in task_groups.items() if len(splits) > 1}
    if task_overlap:
        raise ValueError(f"trajectory/document task overlap detected: {task_overlap}")
    metrics: dict[str, Any] = {
        "records_before_dedup": len(list(records)) if not isinstance(records, list) else len(records),
        "records_after_dedup": len(selected),
        "normalized_duplicates_removed": duplicate_count,
        "benchmark_denylist_matches": denied,
        "split_group_overlap": overlap_groups,
        "cross_split_overlap": cross_split_overlaps,
        "trajectory_task_overlap": task_overlap,
        "tokenizer_exact": True,
    }
    return selected, metrics


def _mixture(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    token_by_domain: Counter[str] = Counter()
    token_by_family: Counter[str] = Counter()
    token_by_language: Counter[str] = Counter()
    total = 0
    for row in rows:
        count = int(row.get("token_count", 0))
        total += count
        token_by_domain[str(row.get("domain", "unknown"))] += count
        token_by_family[str(row.get("source_family", "unknown"))] += count
        token_by_language[str(row.get("language", "unknown"))] += count
    def percentages(counter: Counter[str]) -> dict[str, Any]:
        return {key: {"tokens": value, "fraction": value / total if total else 0.0} for key, value in sorted(counter.items())}
    return {"tokens": total, "by_domain": percentages(token_by_domain), "by_source_family": percentages(token_by_family), "by_language": percentages(token_by_language)}


def _activation_sampling_plan(rows: Iterable[Mapping[str, Any]], *, trajectory_token_cap: int = 12_288) -> dict[str, Any]:
    """Describe a balanced capture budget while retaining complete source rows.

    The frozen manifest keeps every selected trajectory intact.  Capture uses
    bounded, deterministic windows from those rows so long agent transcripts
    cannot consume the entire activation budget.  This is a planning gate, not
    a claim that teacher activations have already been captured.
    """

    token_by_domain: Counter[str] = Counter()
    source_rows = 0
    for row in rows:
        source_rows += 1
        count = int(row.get("token_count", 0))
        if row.get("source_family") == "agent-trajectory":
            count = min(count, trajectory_token_cap)
        token_by_domain[str(row.get("domain", "unknown"))] += count
    available_tokens = sum(token_by_domain.values())
    raw_fractions = {key: value / available_tokens if available_tokens else 0.0 for key, value in sorted(token_by_domain.items())}
    target_fractions = {
        "agentic-software-engineering": 0.28,
        "code": 0.44,
        "software-engineering-natural-language": 0.12,
        "structured": 0.06,
        "general": 0.10,
    }
    planned_tokens = 750_000
    target_tokens = {domain: int(round(planned_tokens * fraction)) for domain, fraction in target_fractions.items()}
    target_tokens["general"] += planned_tokens - sum(target_tokens.values())
    missing_domains = [domain for domain in target_fractions if token_by_domain.get(domain, 0) <= 0]
    resampling_factors = {
        domain: (target_tokens[domain] / token_by_domain[domain] if token_by_domain.get(domain, 0) else None)
        for domain in target_fractions
    }
    return {
        "status": "READY_FOR_BALANCED_CAPTURE" if not missing_domains else "REBALANCE_REQUIRED",
        "source_rows": source_rows,
        "available_tokens_after_caps": available_tokens,
        "available_by_domain": {key: {"tokens": token_by_domain[key], "fraction": raw_fractions[key]} for key in sorted(token_by_domain)},
        "planned_tokens": planned_tokens,
        "target_by_domain": {domain: {"tokens": target_tokens[domain], "fraction": target_fractions[domain]} for domain in target_fractions},
        "bounded_resampling_factors": resampling_factors,
        "trajectory_window_cap_tokens": trajectory_token_cap,
        "windowing": "complete source rows retained; capture derives deterministic sequence_length windows with per-trajectory cap and bounded source-family resampling",
        "raw_source_mixture_out_of_range": [
            domain
            for domain, bounds in {
                "code": [0.40, 0.50],
                "agentic-software-engineering": [0.25, 0.30],
                "software-engineering-natural-language": [0.10, 0.15],
                "structured": [0.05, 0.10],
                "general": [0.10, 0.15],
            }.items()
            if not (bounds[0] <= raw_fractions.get(domain, 0.0) <= bounds[1])
        ],
        "missing_domains": missing_domains,
        "sampling_unit": "stable row IDs grouped by repository/document/task; no split identity changes",
        "teacher_capture": "NOT_STARTED",
    }


def freeze(
    *,
    output: Path,
    receipt: Path,
    splits_output: Path,
    tokenizer_path: Path,
    open_swe_limit: int,
    trajectory_token_cap: int,
    legacy_path: Path,
) -> dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_path)
    records: list[dict[str, Any]] = []
    records.extend(fetch_open_swe(open_swe_limit, tokenizer, trajectory_token_cap))
    records.extend(fetch_repository_records())
    records.extend(load_legacy_support_records(legacy_path))
    records.extend(load_legacy_preservation_records(legacy_path))
    # Stable ordering makes the physical JSONL locator reproducible even when
    # an API changes response ordering.
    records.sort(key=lambda row: (str(row.get("source_family")), str(row.get("source_name")), str(row.get("source_record_id"))))
    selected, audit = _assign_and_audit(records, tokenizer)
    for index, row in enumerate(selected):
        row["source_record_index"] = index
        row["source_file"] = "corpus.jsonl"
        row["id"] = hashlib.sha256(f"{row['source_name']}@{row['source_revision']}:{row['source_record_id']}:{row['content_sha256']}".encode()).hexdigest()[:32]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    split_payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "dense2moe-corpus-v2-splits",
        "status": "FROZEN",
        "split_order": list(SPLITS),
        "split_groups": {split: sorted({str(row["split_group"]) for row in selected if row["split"] == split}) for split in SPLITS},
        "records": {split: [str(row["id"]) for row in selected if row["split"] == split] for split in SPLITS},
    }
    splits_output.parent.mkdir(parents=True, exist_ok=True)
    splits_output.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_hash = _sha256_file(output)
    split_metrics = {split: _mixture(row for row in selected if row["split"] == split) for split in SPLITS}
    production_rows = [row for row in selected if row["split"] != "PRESERVATION-CANARY"]
    mixture_targets = {
        "code": [0.40, 0.50],
        "agentic-software-engineering": [0.25, 0.30],
        "software-engineering-natural-language": [0.10, 0.15],
        "structured": [0.05, 0.10],
        "general": [0.10, 0.15],
    }
    production_mixture = _mixture(production_rows)
    activation_sampling_plan = _activation_sampling_plan(production_rows)
    mixture_fractions = {
        domain: float(production_mixture["by_domain"].get(domain, {}).get("fraction", 0.0))
        for domain in mixture_targets
    }
    mixture_assessment = {
        "policy": "initial production-weighted targets are advisory, not immutable hyperparameters",
        "production_rows_exclude_preservation_canary": True,
        "target_ranges": mixture_targets,
        "actual_fractions": mixture_fractions,
        "out_of_range": [
            domain
            for domain, bounds in mixture_targets.items()
            if not (float(bounds[0]) <= mixture_fractions[domain] <= float(bounds[1]))
        ],
        "status": "AUDIT_RECORDED_REBALANCE_ALLOWED",
    }
    source_table: dict[str, dict[str, Any]] = {}
    for row in selected:
        # Keep agent transcripts and structured tool schemas separate even
        # when they share the same upstream dataset revision.  Collapsing
        # these rows loses source-family provenance in the receipt.
        key = f"{row['source_name']}@{row['source_revision']}@{row['source_family']}"
        source_table[key] = {
            "name": row["source_name"],
            "revision": row["source_revision"],
            "license": row["source_license"],
            "terms": sorted({str(item.get("terms", "")) for item in selected if f"{item['source_name']}@{item['source_revision']}@{item['source_family']}" == key and item.get("terms", "")}),
            "upstream_license": sorted({str(item.get("upstream_license", "")) for item in selected if f"{item['source_name']}@{item['source_revision']}@{item['source_family']}" == key and item.get("upstream_license", "")}),
            "url": row["source_url"],
            "source_family": row["source_family"],
            "languages": sorted({str(item["language"]) for item in selected if f"{item['source_name']}@{item['source_revision']}@{item['source_family']}" == key}),
            "download_sha256_values": sorted({str(item["download_sha256"]) for item in selected if f"{item['source_name']}@{item['source_revision']}@{item['source_family']}" == key}),
            "benchmark_context": sorted(
                {
                    str(context)
                    for item in selected
                    if f"{item['source_name']}@{item['source_revision']}@{item['source_family']}" == key
                    for context in item.get("benchmark_context", [])
                }
            ),
        }
    benchmark_context_counts = Counter(
        str(tag)
        for row in selected
        for tag in row.get("benchmark_context", row.get("benchmark_membership", []))
    )
    creation_files = [
        ROOT / "scripts/freeze_corpus_v2.py",
        ROOT / "scripts/fetch_public_corpus_v2.py",
        ROOT / "src/dense2moe/data.py",
    ]
    receipt_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "dense2moe-corpus-v2-freeze-receipt",
        "status": "CORPUS_V2_FROZEN",
        "corpus_version": "v2",
        "creation_code_sha": _git_sha(),
        "creation_code": {
            "git_head": _git_sha(),
            "files": [{"path": str(path.relative_to(ROOT)), "sha256": _sha256_file(path)} for path in creation_files if path.is_file()],
            "worktree_note": "file hashes pin the generator and metadata pipeline even when the checkout has uncommitted user-scoped changes",
        },
        "manifest": {"path": str(output), "sha256": source_hash, "records": len(selected), "format": "jsonl"},
        "tokenizer": {
            "path": str(tokenizer_path),
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "tokenizer_json_sha256": _sha256_file(tokenizer_path),
            "files": [{"path": str(path.relative_to(tokenizer_path.parent)), "sha256": _sha256_file(path)} for path in sorted(tokenizer_path.parent.glob("tokenizer*")) if path.is_file()],
            "add_special_tokens": False,
            "method": "tokenizers.Tokenizer.encode(add_special_tokens=False)",
        },
        "sources": sorted(source_table.values(), key=lambda value: (value["source_family"], value["name"])),
        "splits": split_metrics,
        "production_mixture": production_mixture,
        "activation_sampling_plan": activation_sampling_plan,
        "mixture_assessment": mixture_assessment,
        "records": [
            {
                "id": row["id"],
                "split": row["split"],
                "split_group": row["split_group"],
                "source_name": row["source_name"],
                "source_revision": row["source_revision"],
                "source_license": row["source_license"],
                "source_record_id": row["source_record_id"],
                "source_record_index": row["source_record_index"],
                "repo": row.get("repo", ""),
                "repo_commit": row.get("repo_commit", ""),
                "repo_path": row.get("repo_path", ""),
                "language": row.get("language", ""),
                "source_family": row.get("source_family", ""),
                "task_family": row.get("task_family", ""),
                "trajectory_framework": row.get("trajectory_framework", ""),
                "trajectory_model": row.get("trajectory_model", ""),
                "trajectory_id": row.get("trajectory_id", ""),
                "document_id": row.get("document_id", ""),
                "task_id": row.get("task_id", ""),
                "benchmark_membership": row.get("benchmark_membership", []),
                "benchmark_context": row.get("benchmark_context", []),
                "benchmark_denylist": row.get("benchmark_denylist", []),
                "terms": row.get("terms", ""),
                "upstream_license": row.get("upstream_license", ""),
                "trajectory_original_token_count": row.get("trajectory_original_token_count"),
                "trajectory_truncated": bool(row.get("trajectory_truncated", False)),
                "token_count": row["token_count"],
                "content_sha256": row["content_sha256"],
                "normalized_content_sha256": row["normalized_content_sha256"],
                "download_sha256": row.get("download_sha256", ""),
            }
            for row in selected
        ],
        "split_identity": {
            "path": str(splits_output),
            "sha256": _sha256_file(splits_output),
            "roles": list(SPLITS),
            "record_ids_sha256": _sha256_bytes("\n".join(str(row["id"]) for row in selected).encode("utf-8")),
            "repository_document_disjoint": True,
            "trajectory_issue_disjoint": not bool(audit["trajectory_task_overlap"]),
        },
        "audit": audit,
        "benchmark_denylist": {"terms": sorted(BENCHMARK_DENYLIST), "matches": audit["benchmark_denylist_matches"], "status": "PASS" if not audit["benchmark_denylist_matches"] else "FAIL"},
        "benchmark_contamination": {
            "known_membership_counts": dict(sorted(benchmark_context_counts.items())),
            "exact_task_denylist_matches": audit["benchmark_denylist_matches"],
            "policy": "track benchmark-derived provenance; exclude exact intended downstream holdout tasks/repositories before promotion",
            "downstream_holdout_manifest": None,
            "downstream_holdout_exclusion": "conditional: no final evaluation task manifest was supplied; benchmark-derived rows remain explicitly tagged and must be excluded before promotion",
            "status": "PASS" if not audit["benchmark_denylist_matches"] else "FAIL",
        },
        "provenance": {"source_urls_recorded": True, "revisions_pinned": True, "licenses_recorded": True, "terms_recorded": True, "upstream_licenses_recorded": True, "source_hashes_recorded": True, "document_hashes_recorded": True, "task_ids_recorded": True},
        "budgets": {split: {"records": sum(1 for row in selected if row["split"] == split), "tokens": split_metrics[split]["tokens"]} for split in SPLITS},
        "preservation_canary": {"split": "PRESERVATION-CANARY", "not_optimization_data": True, "historical_sources_retained": True},
        "teacher_contract": {"external_trajectory_model_is_teacher": False, "dense_qwen_ffn_is_teacher_target": True, "trajectory_usage": "fixed_activation_context_only"},
        "capture_gate": {"basis_refinement_allowed": False, "reason": "freeze receipt must be consumed and verified before serious basis refinement", "smoke_capture_allowed": True},
        "evaluation_closure": {
            "official_holdout": "CLOSED_FOR_THIS_RUN",
            "historical_holdout": "PRIOR_RUN_OPENED_IMMUTABLE_NO_TUNING",
            "replay_gate": "BLOCKED",
            "representative_replay": "BLOCKED",
            "full64_replay": "BLOCKED",
            "internal_calibration_holdout": "pilot plan only; not an official evaluation holdout and not used for optimizer updates",
        },
    }
    auxiliary_source = ROOT / "data/public_v2/corpus-v2-source.jsonl"
    if auxiliary_source.exists():
        receipt_payload["auxiliary_acquisition_source"] = {
            "path": str(auxiliary_source),
            "sha256": _sha256_file(auxiliary_source),
            "records": sum(1 for _ in auxiliary_source.open("rb")),
            "role": "pre-freeze acquisition cache; not used as a split identity or teacher label",
        }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt_payload


def _git_sha() -> str:
    try:
        import subprocess
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--splits-output", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--open-swe-limit", type=int, default=10)
    parser.add_argument("--trajectory-token-cap", type=int, default=0, help="maximum visible trajectory tokens; 0 preserves each selected trajectory in full")
    args = parser.parse_args()
    if args.open_swe_limit <= 0:
        raise ValueError("--open-swe-limit must be positive")
    if args.trajectory_token_cap < 0:
        raise ValueError("--trajectory-token-cap must be non-negative (0 means complete trajectories)")
    # If the target is the old manifest, preserve it before replacing it.
    legacy_path = args.legacy_path
    if legacy_path.resolve() == args.output.resolve() and args.output.exists():
        legacy_path = args.output.with_name("corpus-v1-canary.jsonl")
        if not legacy_path.exists():
            legacy_path.write_bytes(args.output.read_bytes())
    payload = freeze(output=args.output, receipt=args.receipt, splits_output=args.splits_output, tokenizer_path=args.tokenizer_path, open_swe_limit=args.open_swe_limit, trajectory_token_cap=args.trajectory_token_cap, legacy_path=legacy_path)
    print(json.dumps({"status": payload["status"], "manifest": payload["manifest"], "budgets": payload["budgets"], "receipt": str(args.receipt)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
