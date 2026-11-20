"""Wet-mass (liquid density) bookkeeping alongside the preparation ledger.

This is a *separate* accounting from the ledger's solute mass.  The ledger tracks
``mass = const + Σ a_i x_i`` (solute, mg) and deliberately never divides by a
density.  A retention bottle, however, needs the **total wet mass** of the liquid
it holds (solvent + everything dissolved in it) so its centre of gravity can be
placed on a rack.  That number is not recoverable from the solute ledger, so it
gets its own replay.

The rules, straight from the retention-batch spec:

* every original liquid enters with a known wet mass per unit volume (its
  density) and the added solvent likewise;
* the liquid in a container is homogeneous after every step ("already mixed"), so
  a transfer of ``v`` mL out of ``V`` mL removes ``v/V`` of the **remaining total
  wet mass**;
* the density used for the *next* transfer is recomputed from the actual wet mass
  and volume left behind;
* build / transfer / mix / solvent / discard are supported.  Evaporation is not
  part of a retention batch and is rejected rather than guessed at.

All arithmetic is exact :class:`fractions.Fraction`.  Densities are supplied by
container id for the initial build operations and a separate solvent density.
The module can replay a raw operation list or a live ledger's ``log()``; it never
reads or writes the ledger's own tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Sequence

from .rational import ZERO, nonneg, positive, q, qstr

SOLVENT_ORIGIN = "solvent"

_WET_KINDS = ("standard", "sample", "transfer", "mix", "solvent", "discard")


class WetMassError(Exception):
    """A wet-mass replay rule was violated; nothing was guessed."""


@dataclass
class WetContainer:
    cid: str
    volume: Fraction = ZERO
    wet_mass: Fraction = ZERO
    origins: dict[str, tuple[Fraction, Fraction]] = field(default_factory=dict)
    kind: str = "empty"

    @property
    def density(self) -> Fraction | None:
        """Wet mass per unit volume, recomputed from the *actual* remainder."""
        if self.volume == 0:
            return None
        return self.wet_mass / self.volume

    def is_empty(self) -> bool:
        return self.volume == 0

    def to_payload(self) -> dict:
        density = self.density
        return {
            "container": self.cid,
            "kind": self.kind,
            "volume": qstr(self.volume),
            "wet_mass": qstr(self.wet_mass),
            "density": qstr(density) if density is not None else None,
            "origins": [
                {"origin": origin, "volume": qstr(vol), "wet_mass": qstr(mass)}
                for origin, (vol, mass) in sorted(self.origins.items())
            ],
        }


def _op_kind(op: Mapping) -> str:
    kind = op.get("op")
    if kind not in _WET_KINDS:
        if kind == "evaporate":
            raise WetMassError(
                "evaporate is not part of a retention batch; wet mass removed by "
                "evaporation is ambiguous, so it is refused rather than guessed"
            )
        raise WetMassError(f"unsupported operation for wet-mass replay: {kind!r}")
    return kind


def replay_wet_mass(
    ops: Sequence[Mapping],
    densities: Mapping[str, object],
    solvent_density,
) -> dict[str, WetContainer]:
    """Replay ``ops`` into a wet-mass state.

    ``densities`` maps the container id of an initial ``standard``/``sample``
    build to its wet mass per unit volume.  ``solvent_density`` is used for every
    ``solvent`` operation.  The returned mapping is the final container state.
    """
    solvent_rho = nonneg(q(solvent_density, "solvent_density"), "solvent_density")
    if solvent_rho == 0:
        raise WetMassError("solvent_density must be strictly positive")
    init = {cid: q(v, f"density[{cid}]") for cid, v in densities.items()}
    for cid, value in init.items():
        if value <= 0:
            raise WetMassError(f"density[{cid}] must be strictly positive, got {qstr(value)}")

    containers: dict[str, WetContainer] = {}

    def require(cid: str, seq: int) -> WetContainer:
        if cid not in containers:
            raise WetMassError(f"seq {seq}: container {cid!r} does not exist")
        return containers[cid]

    def check_new(seq: int, cid: str) -> None:
        if cid in containers and not containers[cid].is_empty():
            raise WetMassError(f"seq {seq}: container {cid!r} already holds liquid")

    def split(book: Mapping[str, tuple[Fraction, Fraction]], factor: Fraction):
        return {origin: (vol * factor, mass * factor) for origin, (vol, mass) in book.items()}

    for record in ops:
        if "op" in record and "seq" in record and isinstance(record.get("op"), dict):
            seq = int(record["seq"])
            op = record["op"]
        else:  # a raw operation without the ledger's seq wrapper
            seq = 0
            op = record
        kind = _op_kind(op)

        if kind in ("standard", "sample"):
            cid = op["container"]
            check_new(seq, cid)
            if cid not in init:
                raise WetMassError(
                    f"seq {seq}: no wet density supplied for initial liquid in container {cid!r}"
                )
            volume = positive(q(op["volume"], "volume"), "volume")
            wet = volume * init[cid]
            containers[cid] = WetContainer(
                cid=cid, volume=volume, wet_mass=wet,
                origins={cid: (volume, wet)}, kind=kind,
            )

        elif kind == "transfer":
            src = require(op["source"], seq)
            dst_id = op["destination"]
            if not isinstance(dst_id, str) or not dst_id:
                raise WetMassError(f"seq {seq}: destination must be a non-empty string")
            if dst_id == src.cid:
                raise WetMassError(f"seq {seq}: cannot transfer into the source itself")
            volume = positive(q(op["volume"], "volume"), "volume")
            if volume > src.volume:
                raise WetMassError(
                    f"seq {seq}: cannot take {qstr(volume)} mL from {src.cid} "
                    f"holding {qstr(src.volume)} mL"
                )
            if dst_id in containers and not containers[dst_id].is_empty():
                raise WetMassError(f"seq {seq}: destination {dst_id!r} is not empty")
            frac = volume / src.volume
            moved = split(src.origins, frac)
            moved_wet = src.wet_mass * frac
            remain = 1 - frac
            src.origins = split(src.origins, remain)
            src.volume -= volume
            src.wet_mass -= moved_wet
            if src.volume == 0:
                src.origins = {}
                src.wet_mass = ZERO
                src.kind = "empty"
            if dst_id in containers:
                dst = containers[dst_id]
                dst.kind = "mixture"
            else:
                dst = WetContainer(cid=dst_id, kind="mixture")
                containers[dst_id] = dst
            dst.volume = volume
            dst.wet_mass = moved_wet
            dst.origins = moved

        elif kind == "mix":
            sources = op["sources"]
            if not isinstance(sources, list) or not sources:
                raise WetMassError(f"seq {seq}: mix needs a non-empty 'sources' list")
            dst_id = op["destination"]
            if not isinstance(dst_id, str) or not dst_id:
                raise WetMassError(f"seq {seq}: destination must be a non-empty string")
            if dst_id in containers and not containers[dst_id].is_empty():
                raise WetMassError(f"seq {seq}: destination {dst_id!r} is not empty")
            seen: set[str] = set()
            requests: list[tuple[str, Fraction]] = []
            for item in sources:
                cid = item["container"]
                if cid == dst_id:
                    raise WetMassError(f"seq {seq}: destination cannot be a mix source")
                if cid in seen:
                    raise WetMassError(f"seq {seq}: container {cid!r} listed twice")
                seen.add(cid)
                requests.append((cid, positive(q(item["volume"], "volume"), "volume")))

            moved: dict[str, tuple[Fraction, Fraction]] = {}
            total_volume = ZERO
            total_wet = ZERO
            for cid, volume in requests:
                src = require(cid, seq)
                if volume > src.volume:
                    raise WetMassError(
                        f"seq {seq}: cannot take {qstr(volume)} mL from {cid} "
                        f"holding {qstr(src.volume)} mL"
                    )
                frac = volume / src.volume
                total_volume += volume
                total_wet += src.wet_mass * frac
                for origin, (ov, om) in src.origins.items():
                    pv, pm = moved.get(origin, (ZERO, ZERO))
                    moved[origin] = (pv + ov * frac, pm + om * frac)
                src.origins = split(src.origins, 1 - frac)
                src.volume -= volume
                src.wet_mass -= src.wet_mass * frac
                if src.volume == 0:
                    src.origins = {}
                    src.wet_mass = ZERO
                    src.kind = "empty"
            if dst_id in containers:
                dst = containers[dst_id]
                dst.kind = "mixture"
            else:
                dst = WetContainer(cid=dst_id, kind="mixture")
                containers[dst_id] = dst
            dst.volume = total_volume
            dst.wet_mass = total_wet
            dst.origins = moved

        elif kind == "solvent":
            cid = op["container"]
            volume = positive(q(op["volume"], "volume"), "volume")
            if cid not in containers:
                containers[cid] = WetContainer(cid=cid, kind="empty")
            c = containers[cid]
            c.volume += volume
            c.wet_mass += volume * solvent_rho
            pv, pm = c.origins.get(SOLVENT_ORIGIN, (ZERO, ZERO))
            c.origins[SOLVENT_ORIGIN] = (pv + volume, pm + volume * solvent_rho)
            if c.kind == "empty":
                c.kind = "solvent"

        else:  # discard
            c = require(op["container"], seq)
            c.volume = ZERO
            c.wet_mass = ZERO
            c.origins = {}
            c.kind = "empty"
            # NOTE: solute-mass ledger also zeroes; wet mass is a separate book.

    return containers


class WetMassBook:
    """A completed wet-mass replay that can be queried per container."""

    def __init__(
        self,
        containers: Mapping[str, WetContainer],
        densities: Mapping[str, Fraction],
        solvent_density: Fraction,
    ) -> None:
        self.containers = dict(containers)
        self.densities = dict(densities)
        self.solvent_density = solvent_density

    @classmethod
    def replay(cls, ops: Sequence[Mapping], densities: Mapping[str, object], solvent_density) -> "WetMassBook":
        parsed = {cid: q(v, f"density[{cid}]") for cid, v in densities.items()}
        containers = replay_wet_mass(ops, parsed, solvent_density)
        return cls(containers, parsed, q(solvent_density, "solvent_density"))

    @classmethod
    def from_ledger(cls, ledger, densities: Mapping[str, object], solvent_density) -> "WetMassBook":
        """Replay a live ledger's operation log without touching its tables."""
        return cls.replay(ledger.log(), densities, solvent_density)

    def container(self, cid: str) -> WetContainer:
        if cid not in self.containers:
            raise WetMassError(f"container {cid!r} does not exist in the wet-mass book")
        return self.containers[cid]

    def volume(self, cid: str) -> Fraction:
        return self.container(cid).volume

    def wet_mass(self, cid: str) -> Fraction:
        return self.container(cid).wet_mass

    def density(self, cid: str) -> Fraction | None:
        return self.container(cid).density

    def explain(self, cid: str) -> dict:
        return self.container(cid).to_payload()

    def to_payload(self) -> dict:
        return {
            "solvent_density": qstr(self.solvent_density),
            "initial_densities": {cid: qstr(v) for cid, v in sorted(self.densities.items())},
            "containers": [self.containers[cid].to_payload() for cid in sorted(self.containers)],
        }
