"""Retention-bottle geometry, fill state and the load manifest.

A bottle is given as a string of **connected, upright, flat-bottomed inner
chambers** listed from the bottom up.  Chamber ``i`` has a height and a constant
horizontal cross-section; its own base sits at the cumulative height of the
chambers below it.  The bottle's own vertical coordinate ``z`` is zero at the
bottom of the first chamber.

A bottle also carries an **empty-shell mass** and the height of its centre of
gravity (solid shell only).  A *filled* bottle load is the union of the liquid
and the shell:

* liquid fills from the bottom chamber upward; a level exactly on a chamber
  boundary fills that chamber once (never twice);
* a fill that would exceed the total capacity is refused;
* from the fill it reports the liquid level, the total wet mass, and the first
  vertical moment ``Σ m·z`` of liquid + shell (about the bottle bottom).

The *specified* output is a **versioned load manifest**.  A manifest records
every bottle's geometry and shell as well as the derived load, and a correction
to a bottle type produces a new manifest version whose ``previous`` field points
at the old one (with its hash), so the previous layout can be re-examined.  A
manifest is a plain JSON document and can be read back without any of this code's
private state; the rack component consumes only this format.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .rational import ZERO, positive, nonneg, q, qstr

MANIFEST_FORMAT = "prepquant.bottleload/1"


class BottleError(Exception):
    """Invalid geometry, over-capacity fill, or a malformed manifest."""


@dataclass(frozen=True)
class Chamber:
    height: Fraction
    area: Fraction

    def volume(self) -> Fraction:
        return self.height * self.area


@dataclass
class BottleType:
    """A bottle shape plus its empty-shell mass/centre of gravity."""

    name: str
    chambers: tuple[Chamber, ...]
    shell_mass: Fraction
    shell_cg_height: Fraction

    @classmethod
    def build(
        cls,
        name: str,
        chambers: Sequence[Mapping | Sequence],
        shell_mass=0,
        shell_cg_height=0,
    ) -> "BottleType":
        if not isinstance(name, str) or not name:
            raise BottleError("bottle type needs a non-empty name")
        parsed: list[Chamber] = []
        for i, item in enumerate(chambers):
            if isinstance(item, Mapping):
                h = item.get("height")
                a = item.get("area")
            else:
                h, a = item[0], item[1]
            height = positive(q(h, f"chambers[{i}].height"), f"chambers[{i}].height")
            area = positive(q(a, f"chambers[{i}].area"), f"chambers[{i}].area")
            parsed.append(Chamber(height, area))
        if not parsed:
            raise BottleError("a bottle needs at least one chamber")
        sm = nonneg(q(shell_mass, "shell_mass"), "shell_mass")
        scg = nonneg(q(shell_cg_height, "shell_cg_height"), "shell_cg_height")
        return cls(name, tuple(parsed), sm, scg)

    # -- geometry ---------------------------------------------------------
    def total_capacity(self) -> Fraction:
        return sum((c.volume() for c in self.chambers), ZERO)

    def total_height(self) -> Fraction:
        return sum((c.height for c in self.chambers), ZERO)

    def chamber_bases(self) -> list[Fraction]:
        bases: list[Fraction] = []
        z = ZERO
        for c in self.chambers:
            bases.append(z)
            z += c.height
        return bases

    def geometry_payload(self) -> dict:
        bases = self.chamber_bases()
        return {
            "chambers": [
                {
                    "height": qstr(c.height),
                    "area": qstr(c.area),
                    "base": qstr(b),
                    "volume": qstr(c.volume()),
                }
                for c, b in zip(self.chambers, bases)
            ],
            "total_capacity": qstr(self.total_capacity()),
            "total_height": qstr(self.total_height()),
            "shell_mass": qstr(self.shell_mass),
            "shell_cg_height": qstr(self.shell_cg_height),
        }


@dataclass
class FillState:
    """A filled bottle: the liquid book plus the shell, merged into one load."""

    bottle_name: str
    fill_volume: Fraction
    liquid_level: Fraction
    liquid_wet_mass: Fraction
    liquid_moment: Fraction          # Σ m·z of liquid, about bottle bottom
    shell_mass: Fraction
    shell_cg_height: Fraction
    total_mass: Fraction
    total_moment: Fraction           # liquid + shell first vertical moment

    @property
    def cg_height(self) -> Fraction | None:
        if self.total_mass == 0:
            return None
        return self.total_moment / self.total_mass

    def to_payload(self) -> dict:
        return {
            "bottle_type": self.bottle_name,
            "fill_volume": qstr(self.fill_volume),
            "liquid_level": qstr(self.liquid_level),
            "liquid_wet_mass": qstr(self.liquid_wet_mass),
            "liquid_moment": qstr(self.liquid_moment),
            "shell_mass": qstr(self.shell_mass),
            "shell_cg_height": qstr(self.shell_cg_height),
            "total_mass": qstr(self.total_mass),
            "total_moment": qstr(self.total_moment),
            "cg_height": qstr(self.cg_height) if self.cg_height is not None else None,
        }


def fill_bottle(
    bottle: BottleType,
    wet_mass: object,
    wet_density: object,
) -> FillState:
    """Fill ``bottle`` with ``wet_mass`` of liquid of uniform density.

    The fill volume is ``wet_mass / wet_density``.  If it exceeds the capacity
    the fill is refused.  Filling is bottom-up, one chamber at a time; a level
    exactly on a boundary fills that chamber once.
    """
    mass = nonneg(q(wet_mass, "wet_mass"), "wet_mass")
    density = q(wet_density, "wet_density")
    if density <= 0:
        raise BottleError("wet_density must be strictly positive")
    volume = mass / density
    capacity = bottle.total_capacity()
    if volume > capacity:
        raise BottleError(
            f"over capacity: {qstr(volume)} mL requested, bottle {bottle.name!r} holds {qstr(capacity)} mL"
        )

    level = ZERO
    liquid_moment = ZERO
    remaining = volume
    for chamber, base in zip(bottle.chambers, bottle.chamber_bases()):
        if remaining <= 0:
            break
        cap = chamber.volume()
        take = remaining if remaining < cap else cap
        h = take / chamber.area
        # centroid of a prism of height h standing on the chamber base
        liquid_moment += (take * density) * (base + h / 2)
        level = base + h
        remaining -= take
    # ``remaining`` is < every chamber volume by construction; the loop always
    # drains it because total capacity >= volume.

    total_mass = bottle.shell_mass + mass
    total_moment = liquid_moment + bottle.shell_mass * bottle.shell_cg_height
    return FillState(
        bottle_name=bottle.name,
        fill_volume=volume,
        liquid_level=level,
        liquid_wet_mass=mass,
        liquid_moment=liquid_moment,
        shell_mass=bottle.shell_mass,
        shell_cg_height=bottle.shell_cg_height,
        total_mass=total_mass,
        total_moment=total_moment,
    )


# --------------------------------------------------------------------- manifest
def canonical_hash(payload: Mapping) -> str:
    body = {k: v for k, v in payload.items() if k != "manifest_hash"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class BottleLoadManifest:
    version: int
    bottles: dict[str, dict]
    previous: dict | None = None
    note: str | None = None
    manifest_hash: str = ""

    def to_payload(self) -> dict:
        payload = {
            "format": MANIFEST_FORMAT,
            "version": self.version,
            "previous": self.previous,
            "note": self.note,
            "bottles": self.bottles,
        }
        payload["manifest_hash"] = canonical_hash(payload)
        return payload

    def save(self, path) -> dict:
        payload = self.to_payload()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping) -> "BottleLoadManifest":
        """Reconstruct a manifest object from a stored payload (hash verified)."""
        if payload.get("format") != MANIFEST_FORMAT:
            raise BottleError(f"not a {MANIFEST_FORMAT} manifest")
        expected = canonical_hash(payload)
        if payload.get("manifest_hash") != expected:
            raise BottleError(
                f"manifest hash mismatch: stored {payload.get('manifest_hash')!r}, recomputed {expected!r}"
            )
        return cls(
            version=int(payload["version"]),
            bottles={cid: dict(entry) for cid, entry in payload.get("bottles", {}).items()},
            previous=payload.get("previous"),
            note=payload.get("note"),
            manifest_hash=payload["manifest_hash"],
        )

    @classmethod
    def load(cls, path) -> "BottleLoadManifest":
        with open(path, encoding="utf-8") as fh:
            return cls.from_payload(json.load(fh))


def read_manifest(payload: Mapping) -> dict:
    """Read a manifest payload back, verifying its content hash.

    This is the only bridge the rack component uses; it needs no bottle code.
    """
    if payload.get("format") != MANIFEST_FORMAT:
        raise BottleError(f"not a {MANIFEST_FORMAT} manifest")
    expected = canonical_hash(payload)
    if payload.get("manifest_hash") != expected:
        raise BottleError(
            f"manifest hash mismatch: stored {payload.get('manifest_hash')!r}, recomputed {expected!r}"
        )
    bottles: dict[str, dict] = {}
    for cid, entry in payload.get("bottles", {}).items():
        load = entry["load"]
        bottles[cid] = {
            "container": cid,
            "bottle_type": entry["bottle_type"],
            "load": {
                "total_mass": q(load["total_mass"], "load.total_mass"),
                "total_moment": q(load["total_moment"], "load.total_moment"),
                "liquid_wet_mass": q(load["liquid_wet_mass"], "load.liquid_wet_mass"),
                "fill_volume": q(load["fill_volume"], "load.fill_volume"),
                "liquid_level": q(load["liquid_level"], "load.liquid_level"),
            },
        }
    return {
        "format": payload["format"],
        "version": int(payload["version"]),
        "previous": payload.get("previous"),
        "bottles": bottles,
        "manifest_hash": payload["manifest_hash"],
        "raw": payload,
    }


class ManifestBuilder:
    """Build a bottle-load manifest, versioning corrections to bottle types."""

    def __init__(self) -> None:
        self._bottles: dict[str, dict] = {}
        self._version = 0
        self._previous: dict | None = None

    @classmethod
    def resume(cls, previous: BottleLoadManifest) -> "ManifestBuilder":
        """Start a builder whose next ``build`` is a successor of ``previous``."""
        builder = cls()
        builder._bottles = {cid: dict(entry) for cid, entry in previous.bottles.items()}
        builder._version = previous.version
        builder._previous = {
            "version": previous.version,
            "manifest_hash": previous.manifest_hash,
        }
        return builder

    def add(
        self,
        container: str,
        bottle: BottleType,
        wet_mass: object,
        wet_density: object,
    ) -> FillState:
        if not isinstance(container, str) or not container:
            raise BottleError("container id must be a non-empty string")
        state = fill_bottle(bottle, wet_mass, wet_density)
        self._bottles[container] = {
            "container": container,
            "bottle_type": bottle.name,
            "geometry": bottle.geometry_payload(),
            "load": state.to_payload(),
        }
        return state

    def build(self, note: str | None = None) -> BottleLoadManifest:
        self._version += 1
        manifest = BottleLoadManifest(
            version=self._version,
            bottles=dict(self._bottles),
            previous=self._previous,
            note=note,
        )
        # stamp the hash so a successor can point at it
        payload = manifest.to_payload()
        manifest.manifest_hash = payload["manifest_hash"]
        return manifest

    def revise(
        self,
        previous: BottleLoadManifest,
        container: str,
        bottle: BottleType,
        wet_mass: object,
        wet_density: object,
        note: str | None = None,
    ) -> BottleLoadManifest:
        """Replace one bottle's type/fill and produce a successor manifest.

        The successor records ``previous`` = ``{version, manifest_hash}`` so the
        prior layout remains independently readable and auditable.
        """
        resumed = ManifestBuilder.resume(previous)
        self._bottles = resumed._bottles
        self._version = resumed._version
        self._previous = resumed._previous
        self.add(container, bottle, wet_mass, wet_density)
        return self.build(note=note)


def manifest_from_loads(
    loads: Mapping[str, tuple[BottleType, object, object]],
    note: str | None = None,
) -> BottleLoadManifest:
    builder = ManifestBuilder()
    for cid, (bottle, wet_mass, wet_density) in loads.items():
        builder.add(cid, bottle, wet_mass, wet_density)
    return builder.build(note=note)
