"""Advisory knowledge base (kept separate from model predictions; curated with cited sources)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.core.errors import NotFoundError

LANGS = ("en", "hi")
LOCALISED_FIELDS = ("summary", "symptoms", "causes", "prevention", "organic_treatment", "chemical_treatment")


class AdvisoryStore:
    def __init__(self, path: Path):
        raw = path.read_bytes()
        self.version_hash = hashlib.sha256(raw).hexdigest()[:12]
        data = json.loads(raw)
        self.version = data.get("version", "unknown")
        self.disclaimer = data.get("disclaimer", {})
        self.chemical_notice = data.get("chemical_label_notice", {})
        self.entries: dict[str, dict] = {d["id"]: d for d in data["diseases"]}
        self.by_slug: dict[str, str] = {d["slug"]: d["id"] for d in data["diseases"]}

    def resolve(self, key: str) -> dict:
        entry = self.entries.get(key) or self.entries.get(self.by_slug.get(key, ""))
        if entry is None:
            raise NotFoundError(f"Unknown disease id '{key}'.")
        return entry

    def summary(self, entry: dict, lang: str) -> dict:
        return {
            "id": entry["id"],
            "slug": entry["slug"],
            "crop": entry["crop"][lang],
            "name": entry["name"][lang],
            "healthy": entry["healthy"],
            "pathogen_type": entry.get("pathogen_type"),
            "scientific_name": entry.get("scientific_name"),
        }

    def advisory(self, entry: dict, lang: str) -> dict:
        out = self.summary(entry, lang)
        for f in LOCALISED_FIELDS:
            v = entry.get(f)
            if v is None:
                out[f] = None
            elif f == "chemical_treatment":
                out[f] = {"items": v[lang], "active_ingredients": v.get("active_ingredients", [])}
            else:
                out[f] = v[lang]
        out["sources"] = entry.get("sources", [])
        out["disclaimer"] = self.disclaimer.get(lang)
        out["chemical_label_notice"] = self.chemical_notice.get(lang)
        out["language"] = lang
        return out
