"""Experience Storage and Retrieval (ESR) for the ER-QTAO framework.

The experience pool follows the definitions in llm4rail.tex:
* query-agnostic: (tool, failure, diagnosis, strategy)
* query-related: (query, failure, diagnosis, strategy)

The Agent sends each complete trajectory through separate LLM steps for
new-experience summarization and outdated-experience detection; this module
is responsible for retrieval, de-duplication, pruning and storage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

EmbeddingFunction = Callable[[str], list[float]]


def local_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Deterministic offline fallback embedding based on signed feature hashing.

    API embeddings are preferred by Agent. This fallback keeps retrieval usable
    when an embedding endpoint is temporarily unavailable; it is not intended to
    match the semantic quality of a dedicated embedding model.
    """
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    features = list(normalized)
    features.extend(normalized[i : i + 2] for i in range(max(0, len(normalized) - 1)))
    features.extend(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized))
    vector = [0.0] * dimensions
    for feature in features:
        if not feature:
            continue
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        vector[value % dimensions] += 1.0 if value & (1 << 8) else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def extract_json_object(text: str) -> dict:
    """Extract the first valid JSON object, tolerating Markdown code fences."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("The ESR reflector did not return a valid JSON object.")


@dataclass
class Experience:
    kind: str
    failure: str
    diagnosis: str
    strategy: str
    tool: Optional[str] = None
    query: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    embedding: Optional[list[float]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.kind not in {"query_agnostic", "query_related"}:
            raise ValueError(f"Unsupported experience kind: {self.kind}")
        for name in ("failure", "diagnosis", "strategy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Experience field {name} must be a non-empty string.")
            setattr(self, name, value.strip())
        if self.kind == "query_agnostic":
            if not isinstance(self.tool, str) or not self.tool.strip():
                raise ValueError("A query-agnostic experience requires a tool name.")
            self.tool = self.tool.strip()
            # Query-agnostic records must never inherit the current context.
            self.query = None
        else:
            if not isinstance(self.query, str) or not self.query.strip():
                raise ValueError("A query-related experience requires a query.")
            self.query = self.query.strip()
            # Keep the two schemas disjoint even if the LLM emits extra fields.
            self.tool = None

    @property
    def fingerprint(self) -> str:
        payload = {
            "kind": self.kind,
            "tool": self.tool,
            "query": self.query,
            "failure": self.failure,
            "diagnosis": self.diagnosis,
            "strategy": self.strategy,
        }
        normalized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_dict(self, include_embedding: bool = True) -> dict:
        result = {
            "id": self.id,
            "kind": self.kind,
            "tool": self.tool,
            "query": self.query,
            "failure": self.failure,
            "diagnosis": self.diagnosis,
            "strategy": self.strategy,
            "created_at": self.created_at,
        }
        if include_embedding and self.embedding is not None:
            result["embedding"] = self.embedding
        return result

    def to_prompt_dict(self) -> dict:
        return self.to_dict(include_embedding=False)

    @classmethod
    def from_dict(cls, value: dict) -> "Experience":
        return cls(
            id=str(value.get("id") or uuid.uuid4().hex),
            kind=value["kind"],
            tool=value.get("tool"),
            query=value.get("query"),
            failure=value["failure"],
            diagnosis=value["diagnosis"],
            strategy=value["strategy"],
            embedding=value.get("embedding"),
            created_at=value.get("created_at")
            or datetime.now(timezone.utc).isoformat(),
        )


class ExperienceStore:
    """Thread-safe JSON-backed experience pool with top-k cosine retrieval."""

    VERSION = 1

    def __init__(self, path: str | os.PathLike = "service/esr_experiences.json"):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.query_agnostic: list[Experience] = []
        self.query_related: list[Experience] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid ESR storage format in {self.path}")
        experiences = [Experience.from_dict(item) for item in payload.get("experiences", [])]
        self.query_agnostic = [item for item in experiences if item.kind == "query_agnostic"]
        self.query_related = [item for item in experiences if item.kind == "query_related"]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "experiences": [
                item.to_dict() for item in self.query_agnostic + self.query_related
            ],
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary, self.path)

    def retrieve(
        self,
        context_query: str,
        top_k: int = 3,
        embed_fn: Optional[EmbeddingFunction] = None,
    ) -> dict[str, list[Experience]]:
        """Return all query-agnostic and the most similar top-k related items."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        embed = embed_fn or local_embedding
        with self._lock:
            related = list(self.query_related)
            embeddings_changed = False
            if related and top_k:
                query_embedding = [float(value) for value in embed(context_query)]
                scored = []
                for index, experience in enumerate(related):
                    vector = experience.embedding
                    if vector is None or len(vector) != len(query_embedding):
                        vector = [float(value) for value in embed(experience.query or "")]
                        experience.embedding = vector
                        embeddings_changed = True
                    score = cosine_similarity(query_embedding, vector)
                    scored.append((score, -index, experience))
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                selected = [item[2] for item in scored[:top_k]]
            else:
                selected = []
            if embeddings_changed:
                self._save()
            return {
                "query_agnostic": list(self.query_agnostic),
                "query_related": selected,
            }

    @staticmethod
    def format_for_prompt(retrieved: dict[str, list[Experience]]) -> str:
        payload = {
            kind: [experience.to_prompt_dict() for experience in experiences]
            for kind, experiences in retrieved.items()
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def apply_update(
        self,
        update: dict,
        context_query: str,
        embed_fn: Optional[EmbeddingFunction] = None,
    ) -> dict[str, int]:
        """Atomically apply reflector-produced additions and deletions.

        Unknown fields are ignored. Experiences with identical semantic fields
        are de-duplicated, and only IDs present in the pool can be pruned.
        """
        embed = embed_fn or local_embedding
        with self._lock:
            # Work on copies so validation/embedding errors cannot leave a
            # partially updated in-memory pool.
            pools = {
                "query_agnostic": list(self.query_agnostic),
                "query_related": list(self.query_related),
            }
            removed = {"query_agnostic": 0, "query_related": 0}
            for kind, pool in pools.items():
                key = f"outdated_{kind}_ids"
                raw_ids = update.get(key, []) or []
                if not isinstance(raw_ids, list):
                    raise ValueError(f"{key} must be a JSON array")
                ids = {str(value) for value in raw_ids}
                if ids:
                    before = len(pool)
                    pool[:] = [experience for experience in pool if experience.id not in ids]
                    removed[kind] = before - len(pool)

            added = {"query_agnostic": 0, "query_related": 0}
            for kind, pool in pools.items():
                candidates = update.get(kind, []) or []
                if not isinstance(candidates, list):
                    raise ValueError(f"{kind} must be a JSON array")
                fingerprints = {experience.fingerprint for experience in pool}
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    try:
                        experience = Experience(
                            kind=kind,
                            tool=candidate.get("tool"),
                            query=(
                                candidate.get("query") or context_query
                                if kind == "query_related"
                                else None
                            ),
                            failure=candidate["failure"],
                            diagnosis=candidate["diagnosis"],
                            strategy=candidate["strategy"],
                        )
                    except (KeyError, ValueError):
                        continue
                    if not all(
                        [experience.failure, experience.diagnosis, experience.strategy]
                    ):
                        continue
                    if experience.fingerprint in fingerprints:
                        continue
                    if kind == "query_related":
                        experience.embedding = [
                            float(value) for value in embed(experience.query or context_query)
                        ]
                    pool.append(experience)
                    fingerprints.add(experience.fingerprint)
                    added[kind] += 1

            changed = any(added.values()) or any(removed.values())
            if changed:
                self.query_agnostic = pools["query_agnostic"]
                self.query_related = pools["query_related"]
                self._save()
            return {
                "added_query_agnostic": added["query_agnostic"],
                "added_query_related": added["query_related"],
                "removed_query_agnostic": removed["query_agnostic"],
                "removed_query_related": removed["query_related"],
            }

    def all_experiences(self) -> list[Experience]:
        with self._lock:
            return list(self.query_agnostic + self.query_related)
