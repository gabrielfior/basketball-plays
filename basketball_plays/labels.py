"""The user's decisions: cluster names, merges, discards, validation labels, defence labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

VERSION = 1

#: Seed names offered in the page's datalist; the user is free to type anything else.
SEED_NAMES = ("Horns", "1-4 High", "5-Out", "4-Out-1-In", "Box", "Stack", "Zipper", "Floppy",
              "Spain")


@dataclass
class Labels:
    """`clusters` maps a cluster id to a set name, to `"discard"`, or to `"merge:<other id>"`.

    `validation` and `defense` map a record key (`"<game_id>:<period>:<index>"`) to the user's
    own judgement: a set name (or `"other"`/`"unclear"`) and `man`/`zone`/`unclear`.
    """

    clusters: dict[str, str] = field(default_factory=dict)
    validation: dict[str, str] = field(default_factory=dict)
    defense: dict[str, str] = field(default_factory=dict)
    version: int = VERSION


def load(path: str | Path) -> Labels:
    p = Path(path)
    if not p.exists():
        return Labels()
    d = json.loads(p.read_text())
    return Labels(clusters=d.get("clusters", {}), validation=d.get("validation", {}),
                  defense=d.get("defense", {}), version=d.get("version", VERSION))


def save(labels: Labels, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(labels), indent=2, sort_keys=True) + "\n")


def resolve(labels: Labels, cluster_id: str, _depth: int = 0) -> str | None:
    """The set name a cluster ends up with, following `merge:` chains.

    `None` when the cluster is unnamed, discarded, or the chain is circular (depth capped).
    """
    val = labels.clusters.get(str(cluster_id))
    if val is None or val == "discard" or _depth > 10:
        return None
    if val.startswith("merge:"):
        return resolve(labels, val[len("merge:"):], _depth + 1)
    return val


def apply(labels: Labels, cluster_labels: dict[str, int]) -> dict[str, str | None]:
    """Record key -> set name for every record, `None` where the cluster has no usable name."""
    return {key: resolve(labels, str(c)) for key, c in cluster_labels.items()}
