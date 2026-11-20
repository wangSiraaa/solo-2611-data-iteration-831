"""PREPQUANT — exact dilution-ledger and sample-quantitation package.

Two independent components share only rational arithmetic and an exported product
format:

* :class:`prepquant.ledger.Ledger` — ordered, atomic, replayable preparation log
  with provenance books and content-addressed product versions.
* :class:`prepquant.vault.Vault` — measurements pinned to exported products and
  jointly solved exact interval constraints.

All numbers are :class:`fractions.Fraction`; nothing is ever rounded.
"""

from .rational import q, qstr, ZERO
from .expr import LinExpr
from .ledger import Ledger, LedgerError, Product, product_from_payload
from .lp import Infeasible, Unbounded, SolverError
from .vault import Vault, VaultError
from .wetmass import WetMassBook, WetMassError, replay_wet_mass
from .bottles import (
    BottleError,
    BottleLoadManifest,
    BottleType,
    FillState,
    ManifestBuilder,
    fill_bottle,
    manifest_from_loads,
    read_manifest,
)
from .rack import (
    BottlePlacement,
    Condition,
    Hole,
    Rack,
    RackError,
    RackSolution,
    Weight,
    placements_from_manifest,
    solve_across_racks,
    solve_rack,
)

__all__ = [
    "q",
    "qstr",
    "ZERO",
    "LinExpr",
    "Ledger",
    "LedgerError",
    "Product",
    "product_from_payload",
    "Vault",
    "VaultError",
    "Infeasible",
    "Unbounded",
    "SolverError",
    # wet mass (entry point 1, part a)
    "WetMassBook",
    "WetMassError",
    "replay_wet_mass",
    # bottle load manifest (entry point 1, part b)
    "BottleError",
    "BottleLoadManifest",
    "BottleType",
    "FillState",
    "ManifestBuilder",
    "fill_bottle",
    "manifest_from_loads",
    "read_manifest",
    # rack counterweight (entry point 2)
    "BottlePlacement",
    "Condition",
    "Hole",
    "Rack",
    "RackError",
    "RackSolution",
    "Weight",
    "placements_from_manifest",
    "solve_across_racks",
    "solve_rack",
]
