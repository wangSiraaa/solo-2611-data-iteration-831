from fractions import Fraction as F

import pytest

from prepquant.bottles import (
    BottleError,
    BottleType,
    ManifestBuilder,
    fill_bottle,
    manifest_from_loads,
    read_manifest,
)
from prepquant.wetmass import WetMassBook, WetMassError


# ----------------------------------------------------------------- wet mass
def test_proportional_transfer_uses_remaining_total():
    ops = [
        {"op": "sample", "container": "SRC", "volume": "10", "sample": "s1"},
        {"op": "transfer", "source": "SRC", "volume": "4", "destination": "A"},
        {"op": "transfer", "source": "SRC", "volume": "3", "destination": "B"},
    ]
    # up to (and including) the first transfer: 12 wet in 10 mL,
    # take 4 mL -> 4/10 of 12 = 24/5; leave 36/5 in 6 mL.
    first = WetMassBook.replay(ops[:2], {"SRC": "6/5"}, "1")
    assert first.wet_mass("A") == F(24, 5)
    assert first.container("SRC").wet_mass == F(36, 5)
    # full replay: next transfer is 3 of the *remaining* 6 mL -> half of 36/5
    book = WetMassBook.replay(ops, {"SRC": "6/5"}, "1")
    assert book.wet_mass("A") == F(24, 5)
    assert book.wet_mass("B") == F(18, 5)
    assert book.container("SRC").wet_mass == F(18, 5)   # 36/5 - 18/5
    assert book.density("SRC") == F(6, 5)
    assert book.volume("SRC") == 3


def test_density_recomputed_after_solvent():
    ops = [
        {"op": "sample", "container": "S", "volume": "4", "sample": "s1"},   # rho 2 -> wet 8
        {"op": "transfer", "source": "S", "volume": "2", "destination": "D"},  # 2 mL, wet 4
        {"op": "solvent", "container": "D", "volume": "2"},                    # +2 wet -> 4 mL, 6 wet
        {"op": "transfer", "source": "D", "volume": "1", "destination": "E"},  # 1/4 of 6 = 3/2
        {"op": "solvent", "container": "E", "volume": "1"},                    # 2 mL, 5/2 wet
    ]
    book = WetMassBook.replay(ops, {"S": "2"}, "1")
    assert book.container("D").wet_mass == F(9, 2)     # 6 - 3/2
    assert book.density("D") == F(3, 2)                # 4.5 / 3
    assert book.container("E").wet_mass == F(5, 2)     # 3/2 + 1
    assert book.density("E") == F(5, 4)                # 2.5 / 2


def test_mix_sums_wet_mass_and_origins():
    ops = [
        {"op": "sample", "container": "A", "volume": "2", "sample": "s1"},  # rho 3 -> 6
        {"op": "sample", "container": "B", "volume": "4", "sample": "s2"},  # rho 1 -> 4
        {"op": "mix", "sources": [{"container": "A", "volume": "1"}, {"container": "B", "volume": "2"}], "destination": "M"},
    ]
    book = WetMassBook.replay(ops, {"A": "3", "B": "1"}, "1")
    # 1 mL of A (rho 3) = 3 wet; 2 mL of B (rho 1) = 2 wet
    assert book.wet_mass("M") == 5
    assert book.volume("M") == 3
    assert book.density("M") == F(5, 3)
    assert book.container("A").wet_mass == 3 and book.container("A").volume == 1
    assert book.container("B").wet_mass == 2 and book.container("B").volume == 2


def test_discard_zeroes_wet_mass():
    ops = [
        {"op": "sample", "container": "S", "volume": "4", "sample": "s1"},
        {"op": "discard", "container": "S"},
    ]
    book = WetMassBook.replay(ops, {"S": "1"}, "1")
    assert book.wet_mass("S") == 0 and book.volume("S") == 0


def test_over_draw_and_missing_density_rejected():
    with pytest.raises(WetMassError):
        WetMassBook.replay(
            [{"op": "sample", "container": "S", "volume": "2", "sample": "s1"},
             {"op": "transfer", "source": "S", "volume": "5", "destination": "D"}],
            {"S": "1"}, "1",
        )
    with pytest.raises(WetMassError):
        WetMassBook.replay([{"op": "sample", "container": "S", "volume": "2", "sample": "s1"}], {}, "1")


def test_evaporate_rejected_in_wet_batch():
    with pytest.raises(WetMassError):
        WetMassBook.replay(
            [{"op": "sample", "container": "S", "volume": "2", "sample": "s1"},
             {"op": "evaporate", "container": "S", "to": "1"}],
            {"S": "1"}, "1",
        )


def test_replay_from_ledger_log():
    import os, tempfile
    from prepquant.ledger import Ledger

    d = tempfile.mkdtemp()
    L = Ledger(os.path.join(d, "l.db"))
    L.apply_batch([
        {"op": "sample", "container": "R", "volume": "4", "sample": "s1"},
        {"op": "transfer", "source": "R", "volume": "2", "destination": "D"},
    ])
    book = WetMassBook.from_ledger(L, {"R": "3/2"}, "1")
    assert book.wet_mass("R") == 3 and book.wet_mass("D") == 3
    assert book.density("D") == F(3, 2)
    L.close()


# ------------------------------------------------------------------ bottles
def test_single_chamber_fill_level_mass_moment():
    b = BottleType.build("tall", [{"height": "8", "area": "1/2"}])
    s = fill_bottle(b, wet_mass=2, wet_density=1)
    assert s.fill_volume == 2
    assert s.liquid_level == 4
    assert s.liquid_wet_mass == 2
    assert s.liquid_moment == 4          # 2 kg at cg height 2
    assert s.total_moment == 4           # shell mass zero
    assert s.cg_height == 2


def test_wide_bottle_lowers_centre_of_gravity():
    b = BottleType.build("wide", [{"height": "8", "area": "1"}])
    s = fill_bottle(b, wet_mass=2, wet_density=1)
    assert s.liquid_level == 2
    assert s.liquid_moment == 2
    assert s.cg_height == 1


def test_fill_across_two_chambers_once_per_boundary():
    # first chamber volume 2 (h1, area2); second chamber volume 2.5 (h5, area1/2)
    b = BottleType.build("cross", [{"height": "1", "area": "2"}, {"height": "5", "area": "1/2"}])
    s = fill_bottle(b, wet_mass=3, wet_density=1)
    assert s.liquid_level == 3           # first chamber full to z=1, then 1 mL / 0.5 = 2
    # moment: first chamber 2 kg at z=0.5 -> 1; second 1 kg at z=1+1 = 2 -> 2; total 3
    assert s.liquid_moment == 3
    assert s.fill_volume == 3


def test_boundary_level_fills_chamber_once():
    b = BottleType.build("two", [{"height": "1", "area": "2"}, {"height": "3", "area": "1"}])
    # exactly the first chamber's volume -> level sits on the boundary, top once
    s = fill_bottle(b, wet_mass=2, wet_density=1)
    assert s.liquid_level == 1
    assert s.liquid_moment == 1          # 2 kg at cg height 0.5


def test_over_capacity_rejected():
    b = BottleType.build("small", [{"height": "1", "area": "1"}])
    with pytest.raises(BottleError):
        fill_bottle(b, wet_mass=2, wet_density=1)


def test_shell_mass_and_cg_are_included():
    b = BottleType.build("shell", [{"height": "4", "area": "1/2"}], shell_mass=3, shell_cg_height=2)
    s = fill_bottle(b, wet_mass=2, wet_density=1)   # 2 mL, level 4, liquid moment 4
    assert s.total_mass == 5
    assert s.total_moment == 4 + 3 * 2             # 10
    assert s.cg_height == 2


def test_two_bottles_same_fill_different_shape():
    # the archive's two retention bottles: same liquid wet mass, one tall one wide
    tall = fill_bottle(BottleType.build("tall", [{"height": "8", "area": "1/2"}]), 2, 1)
    wide = fill_bottle(BottleType.build("wide", [{"height": "8", "area": "1"}]), 2, 1)
    assert tall.total_mass == wide.total_mass == 2
    assert tall.liquid_level == 4 and wide.liquid_level == 2
    assert tall.liquid_moment == 4 and wide.liquid_moment == 2


# ------------------------------------------------------------------ manifest
def test_manifest_roundtrip_and_hash():
    b = BottleType.build("tall", [{"height": "8", "area": "1/2"}])
    m = manifest_from_loads({"RAW": (b, 2, 1)})
    payload = m.to_payload()
    parsed = read_manifest(payload)
    assert parsed["version"] == 1
    assert parsed["bottles"]["RAW"]["load"]["total_moment"] == 4
    assert parsed["manifest_hash"] == payload["manifest_hash"]
    # tampering is detected
    bad = dict(payload)
    bad["bottles"] = dict(payload["bottles"])
    bad["bottles"]["RAW"] = dict(payload["bottles"]["RAW"])
    bad["bottles"]["RAW"]["load"] = dict(payload["bottles"]["RAW"]["load"])
    bad["bottles"]["RAW"]["load"]["total_moment"] = "99"
    with pytest.raises(BottleError):
        read_manifest(bad)


def test_manifest_revision_links_previous():
    builder = ManifestBuilder()
    tall = BottleType.build("tall", [{"height": "8", "area": "1/2"}])
    wide = BottleType.build("wide", [{"height": "8", "area": "1"}])
    v1 = builder.build()
    builder.add("RAW", tall, 2, 1)
    v1 = builder.build()
    v2 = builder.revise(v1, "RAW", wide, 2, 1, note="bottle type corrected")
    assert v2.version == v1.version + 1
    assert v2.previous == {"version": v1.version, "manifest_hash": v1.manifest_hash}
    # the previous layout is still independently readable
    assert v1.to_payload()["bottles"]["RAW"]["load"]["total_moment"] == "4"
    assert v2.to_payload()["bottles"]["RAW"]["load"]["total_moment"] == "2"
    # the successor's hash must change
    assert v1.manifest_hash != v2.manifest_hash
