"""Command line interface: ``python -m prepquant.cli``.

Examples
--------
    python -m prepquant.cli --ledger lab.db --vault lab.vault.db init
    python -m prepquant.cli --ledger lab.db --vault lab.vault.db prep ops.json
    python -m prepquant.cli --ledger lab.db explain M
    python -m prepquant.cli --ledger lab.db export M --out m.json
    python -m prepquant.cli --ledger lab.db --vault lab.vault.db measure --name r1 \
        --product-file m.json --observed 44.6 --uncertainty 0.1
    python -m prepquant.cli --ledger lab.db --vault lab.vault.db analyze a1 \
        --measurement r1 --bound s1:0:100
    python -m prepquant.cli --ledger lab.db --vault lab.vault.db show-analysis a1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bottles import (
    BottleError,
    BottleLoadManifest,
    BottleType,
    ManifestBuilder,
    read_manifest,
)
from .ledger import Ledger, LedgerError
from .rack import (
    BottlePlacement,
    Condition,
    Rack,
    RackError,
    Weight,
    placements_from_manifest,
    solve_across_racks,
    solve_rack,
)
from .rational import q, qstr
from .vault import Vault, VaultError
from .wetmass import WetMassBook, WetMassError


def _load_ops(path: str | None) -> list[dict]:
    if path in (None, "-"):
        text = sys.stdin.read()
        source = "stdin"
    else:
        text = Path(path).read_text(encoding="utf-8")
        source = path
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"cannot parse operations from {source}: {exc}")
    if isinstance(data, dict) and "operations" in data:
        data = data["operations"]
    if not isinstance(data, list):
        raise SystemExit("operations file must be a JSON list or {\"operations\": [...]}")
    return data


def _parse_bound(text: str) -> tuple[str, str]:
    parts = text.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--bound must be NAME:LOW:HIGH, got {text!r}")
    return parts[0], (parts[1], parts[2])


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False, default=str))


def _print_explain(e: dict) -> None:
    print(f"container {e['container']} ({e['kind']}, version {e['version']})")
    print(f"  volume        {e['volume']} mL")
    print(f"  solute mass   {e['mass']} mg")
    print(f"  conc.         {e['concentration']} mg/mL")
    if e["origins"]:
        print("  provenance (origin -> volume in mL, mass in mg):")
        for o in e["origins"]:
            print(f"    {o['origin']:<12} {o['volume']:>10} mL   {o['mass']}")
    else:
        print("  provenance    (empty)")
    print(f"  unknown samples: {', '.join(e['samples']) or '(none)'}")


def _print_analysis(r: dict) -> None:
    print(f"analysis {r['name']}  status: {r['status']}")
    if r["status"] == "contradiction":
        print("  no concentration vector satisfies every reading and bound.")
        for c in r.get("individual_conflicts", []):
            print(f"  individually impossible: {c['measurement']}  {c['reason']}")
            print(f"    attainable mass {c['attainable_mass']} vs reading mass {c['reading_mass_interval']}")
        return
    print(f"  feasibility witness (NOT an estimate; any point in the ranges is valid): {r['witness_point']}")
    print(f"  unique solution: {'yes' if r['unique'] else 'no (ranges below are exact)'}")
    for u, e in r["endpoints"].items():
        flag = "unique" if e["unique"] else "range"
        print(f"  {u}: [{e['min']}, {e['max']}] mg/mL   ({flag})")
        if not e["unique"]:
            print(f"      at min other samples: {r['endpoint_points'][u]['min']}")
            print(f"      at max other samples: {r['endpoint_points'][u]['max']}")


def _load_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON from {path}: {exc}")


def _bottle_from_spec(spec: dict) -> BottleType:
    return BottleType.build(
        spec["name"],
        spec["chambers"],
        shell_mass=spec.get("shell_mass", 0),
        shell_cg_height=spec.get("shell_cg_height", 0),
    )


def _run_bottle_load(args) -> int:
    spec = _load_json(args.spec)
    operations = spec.get("operations", [])
    densities = spec.get("densities", {})
    solvent_density = spec.get("solvent_density", "1")
    fills = spec["fills"]  # {container: {bottle: <spec>, wet_mass, wet_density?}}

    book = WetMassBook.replay(operations, densities, solvent_density)
    if args.previous:
        builder = ManifestBuilder.resume(BottleLoadManifest.load(args.previous))
    else:
        builder = ManifestBuilder()
    for cid, fill in fills.items():
        bottle = _bottle_from_spec(fill["bottle"])
        wet_mass = fill.get("wet_mass")
        wet_density = fill.get("wet_density")
        if wet_mass is None:
            wet_mass = book.wet_mass(cid)
        if wet_density is None:
            wet_density = book.density(cid) or "1"
        builder.add(cid, bottle, wet_mass, wet_density)
    manifest = builder.build(note=args.note)
    payload = manifest.save(args.out)
    print(
        f"manifest v{payload['version']} ({payload['manifest_hash']}) written to {args.out}: "
        f"{len(payload['bottles'])} bottle(s)"
    )
    _print(payload)
    return 0


def _run_manifest_show(args) -> int:
    payload = _load_json(args.file)
    parsed = read_manifest(payload)
    view = {
        "format": parsed["format"],
        "version": parsed["version"],
        "previous": parsed["previous"],
        "manifest_hash": parsed["manifest_hash"],
        "bottles": {
            cid: {
                "bottle_type": entry["bottle_type"],
                "total_mass": qstr(entry["load"]["total_mass"]),
                "total_moment": qstr(entry["load"]["total_moment"]),
                "liquid_wet_mass": qstr(entry["load"]["liquid_wet_mass"]),
                "liquid_level": qstr(entry["load"]["liquid_level"]),
            }
            for cid, entry in parsed["bottles"].items()
        },
    }
    _print(view)
    return 0


def _run_rack_solve(args) -> int:
    spec = _load_json(args.spec)
    if "racks" in spec:
        rack_specs = spec["racks"]
    else:
        rack_specs = [spec]  # single-rack convenience form
    racks = [
        Rack.build(
            r.get("name", f"rack{i+1}"),
            r["holes"],
            frame_mass=r.get("frame_mass", 0),
            frame_moment_x=r.get("frame_moment_x", 0),
            frame_moment_z=r.get("frame_moment_z", 0),
        )
        for i, r in enumerate(rack_specs)
    ]
    if "manifest" in spec:
        manifest_path = Path(spec["manifest"])
        if not manifest_path.is_absolute() and not manifest_path.exists():
            # resolve relative to the spec file for convenience
            candidate = Path(args.spec).resolve().parent / manifest_path
            if candidate.exists():
                manifest_path = candidate
        manifest_payload = _load_json(str(manifest_path))
        manifest = read_manifest(manifest_payload)
        placements = placements_from_manifest(manifest, spec["positions"])
    else:
        placements = [BottlePlacement.build(**p) for p in spec["placements"]]
    weights = [
        Weight(w["name"], q(w["mass"], "weight.mass"), tuple(w.get("allowed_holes", ())))
        for w in spec.get("weights", [])
    ]
    conditions = [
        Condition.build(c["name"], c["k_lo"], c["k_hi"], c["a"], c["b"])
        for c in spec["conditions"]
    ]
    if len(racks) == 1:
        _print(solve_rack(racks[0], placements, weights, conditions).to_payload())
    else:
        _print(solve_across_racks(racks, placements, weights, conditions))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prepquant", description="Exact dilution ledger and quantitation vault")
    p.add_argument("--ledger", default="ledger.db", help="preparation ledger SQLite file")
    p.add_argument("--vault", default="vault.db", help="quantitation vault SQLite file")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create/reopen both databases")

    p_prep = sub.add_parser("prep", help="apply an ordered atomic operation batch (JSON file or '-')")
    p_prep.add_argument("file", help="path to JSON operations, or '-' for stdin")

    p_correct = sub.add_parser("correct", help="replace one step and replay every later step")
    p_correct.add_argument("seq", type=int)
    p_correct.add_argument("file")
    p_correct.add_argument("--note")

    p_explain = sub.add_parser("explain", help="volume, mass expression and provenance of one container")
    p_explain.add_argument("container")
    p_explain.add_argument("--json", action="store_true")

    p_export = sub.add_parser("export", help="export the current product version of a container")
    p_export.add_argument("container")
    p_export.add_argument("--out", help="write JSON here instead of stdout")

    sub.add_parser("containers", help="list containers")
    p_log = sub.add_parser("log", help="show the operation log")
    p_log.add_argument("--json", action="store_true")
    sub.add_parser("corrections", help="show correction audit trail")
    sub.add_parser("consumed", help="list product versions consumed by analyses")

    p_measure = sub.add_parser("measure", help="import one measurement against an exported product")
    p_measure.add_argument("--name", required=True)
    src = p_measure.add_mutually_exclusive_group(required=True)
    src.add_argument("--product-file", help="JSON product payload (external format)")
    src.add_argument("--product-id", help="product id currently in the ledger")
    src.add_argument("--container", help="export the current product of this container")
    p_measure.add_argument("--observed", required=True, help="mg/mL, exact rational")
    p_measure.add_argument("--uncertainty", required=True, help="mg/mL half-width, exact rational")
    sub.add_parser("measurements", help="list imported measurements")

    p_analyze = sub.add_parser("analyze", help="jointly solve a set of measurements")
    p_analyze.add_argument("name")
    p_analyze.add_argument("--measurement", action="append", required=True, dest="measurements")
    p_analyze.add_argument("--bound", action="append", required=True, dest="bounds", metavar="NAME:LOW:HIGH")
    p_analyze.add_argument("--no-consume", action="store_true", help="do not pin/consume products (debugging)")

    p_show = sub.add_parser("show-analysis", help="show a stored analysis result")
    p_show.add_argument("name")
    sub.add_parser("analyses", help="list stored analyses")

    # -- offline retention model, entry point 1: bottle load manifest --------
    p_load = sub.add_parser(
        "bottle-load",
        help="replay wet mass and emit/revise a versioned bottle-load manifest",
    )
    p_load.add_argument("spec", help="JSON load spec (operations, densities, bottles)")
    p_load.add_argument("--out", required=True, help="write the manifest JSON here")
    p_load.add_argument("--previous", help="previous manifest JSON to revise")
    p_load.add_argument("--note", help="note recorded on the new manifest version")

    p_mshow = sub.add_parser("manifest-show", help="read a bottle-load manifest back")
    p_mshow.add_argument("file", help="manifest JSON to read")

    # -- offline retention model, entry point 2: rack counterweights ---------
    p_rack = sub.add_parser(
        "rack-solve",
        help="find the minimum-added-mass counterweight layout for a rack",
    )
    p_rack.add_argument("spec", help="JSON rack spec (rack, placements/manifest, weights, conditions)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ledger = Ledger(args.ledger)
    except LedgerError as exc:
        raise SystemExit(f"ledger error: {exc}")
    try:
        vault = Vault(args.vault)
    except VaultError as exc:
        raise SystemExit(f"vault error: {exc}")

    try:
        if args.command == "init":
            print(f"ledger {args.ledger}: {len(ledger.ops)} steps, {len(ledger.containers)} containers")
            print(f"vault  {args.vault}: {len(vault.list_measurements())} measurements, "
                  f"{len(vault.list_analyses())} analyses")
            return 0

        if args.command == "prep":
            seqs = ledger.apply_batch(_load_ops(args.file))
            print(f"applied {len(seqs)} step(s): seq {seqs[0]}..{seqs[-1]}" if seqs else "nothing to do")
            return 0

        if args.command == "correct":
            seqs = ledger.correct(args.seq, _load_ops(args.file), note=args.note)
            print(f"replayed ledger: {len(seqs)} step(s) total, correction at seq {args.seq}")
            return 0

        if args.command == "explain":
            e = ledger.explain(args.container)
            _print(e) if args.json else _print_explain(e)
            return 0

        if args.command == "export":
            payload = ledger.export_product(args.container)
            text = json.dumps(payload, indent=2, ensure_ascii=False)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
                print(f"wrote {args.out} ({payload['product_id']})")
            else:
                print(text)
            return 0

        if args.command == "containers":
            rows = []
            for c in ledger.list_containers():
                rows.append(
                    {
                        "container": c.cid,
                        "kind": c.kind,
                        "volume": qstr(c.volume),
                        "mass": c.mass.formatted(),
                        "version": c.version,
                    }
                )
            _print(rows)
            return 0

        if args.command == "log":
            log = ledger.log()
            if args.json:
                _print(log)
            else:
                for r in log:
                    print(f"{r['seq']:>3}  {r['kind']:<10} {json.dumps(r['op'], sort_keys=True, ensure_ascii=False)}")
            return 0

        if args.command == "corrections":
            _print(ledger.corrections())
            return 0

        if args.command == "consumed":
            _print(ledger.consumed_products())
            return 0

        if args.command == "measure":
            if args.product_file:
                product = json.loads(Path(args.product_file).read_text(encoding="utf-8"))
            elif args.product_id:
                product = ledger.product(args.product_id).to_payload()
            else:
                product = ledger.export_product(args.container)
            m = vault.add_measurement(args.name, product, args.observed, args.uncertainty)
            print(
                f"measurement {m['name']} -> product {m['product_id']}  "
                f"conc {m['conc_expr'].formatted()}  interval "
                f"[{qstr(m['interval'][0])}, {qstr(m['interval'][1])}]"
            )
            return 0

        if args.command == "measurements":
            _print(vault.list_measurements())
            return 0

        if args.command == "analyze":
            bounds = {}
            for item in args.bounds:
                name, limits = _parse_bound(item)
                if name in bounds and bounds[name] != limits:
                    raise SystemExit(f"duplicate --bound for {name}")
                bounds[name] = limits
            result = vault.analyze(
                args.name,
                args.measurements,
                bounds,
                ledger=None if args.no_consume else ledger,
                record_consumption=not args.no_consume,
            )
            _print_analysis(result)
            return 0

        if args.command == "show-analysis":
            _print_analysis(vault.analysis(args.name))
            return 0

        if args.command == "analyses":
            _print(vault.list_analyses())
            return 0

        if args.command == "bottle-load":
            return _run_bottle_load(args)

        if args.command == "manifest-show":
            return _run_manifest_show(args)

        if args.command == "rack-solve":
            return _run_rack_solve(args)
    except (LedgerError, VaultError, WetMassError, BottleError, RackError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise SystemExit(f"unhandled command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
