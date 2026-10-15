"""Trusted CPU Structure scorer; writes one exclusive report, never admission.

Usage: python -B score_outputs.py --update 20|100|300 [--seed 1701|1702]
Without --seed, BOTH training seeds must be complete before any report is saved.
Reports go to JOB/scores/g1_a[_seedNNNN]_plusUUU.json via paths.save().

Schema assumptions, from the local producers (no datasets are bundled here):
* make_eval_packets.py supplies rows and the VERIFIER_FREEZE.json hash.
* infer_structure.py supplies per-unit and root COMPLETE.json receipts, and
  predicted_Q/predicted_S/structure_noise/checkpoint/no_target_input in each PT.
* LABEL_MANIFEST.json maps scene IDs to frozen NPZ files containing target,
  original masks, and shared_closed_walls for open variants.
* LEGACY_RAW_REGRESSION.json has rows with arm/scene/value/raw_path/raw_sha256/
  target_path/target_sha256. Only its four approved development targets are read.
* HISTORICAL_ROWS.json indexes the already verified historical outputs. MIX0
  +300 is recomputed against frozen labels and INPUT_HASHES, not trusted as a
  supplied metric summary. The legacy audit is trusted bootstrap metadata;
  its own hash and all consumed raw/target hashes are recorded in this report.

Both full evaluations (+100/+300) use the SAME frozen MIX0 +300 reference and
thresholds. Historical +100 has only four main units and cannot supply a full
48-unit baseline. +20 compares only its packet subset, with NO native gate.
Challenges are scored separately, never included in main means or gates.
Undefined diagnostic ratios remain null with explicit coverage counts; primary
means never filter missing values. No heldout, unfiltered dataset manifest,
original sealed test, model, candidate, trainer, or Gaussian decoder is loaded.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np
import torch

from paths import DATA, JOB, NATIVE, save, sha
import verifier_core as verifier


FIELDS = (
    "free_air_wrong_fraction", "overhead_column_fraction", "required_surface_recall",
    "non_ground_native_IoU", "protected_region_symmetric_difference_fraction",
)
VARIANTS = ("closed", "open3m", "open4m")
SEEDS = (1701, 1702)
CANDIDATE = 'g1_a'
HISTORICAL_ARMS = ("MIX0", "O1A", "O1B", "O3A", "O3B", "C4A", "C4B")
MASKS = ("free_air", "overhead", "required_surface", "closed_closure", "roi",
         "protected", "non_ground", "unknown_geometry", "eroded_xy")
ROOF_FIELDS = ("exact_recall", "same_xy_any_non_ground_fraction", "within_0p5m_fraction",
               "within_1m_fraction", "signed_nearest_offset_median_m",
               "abs_nearest_offset_median_m", "missing_entire_column_fraction")
PAIR_FIELDS = ("shared_closed_recall", "shared_open_recall", "shared_both_recalled",
               "intended_change_recall", "intended_change_precision",
               "directionally_correct_changed_fraction", "protected_region_change_fraction",
               "unrelated_non_ground_change_fraction")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, where):
    require(isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)), f"{where}: expected a number")
    require(math.isfinite(float(value)), f"{where}: nonfinite value")
    return float(value)


def finite_tree(value, where="value"):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            finite_tree(item, f"{where}[{i}]")
    elif isinstance(value, (float, np.floating)):
        number(value, where)


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


class Inputs:
    """Hash every consumed file; check only needed sealed files, never heldout."""

    def __init__(self):
        self.hashes = {}
        self.checked_frozen = set()
        self.freeze_path = JOB / "VERIFIER_FREEZE.json"
        self.freeze = self.json(self.freeze_path)
        self.freeze_sha256 = self.hashes[str(self.freeze_path)]
        self.files = self.freeze["files"]
        require(isinstance(self.files, dict), "invalid verifier freeze")
        self.frozen(Path(verifier.__file__))
        self.frozen(JOB / "paths.py")

    def file(self, path, expected=None):
        path = Path(path)
        require(path.is_file(), f"missing file: {path}")
        if expected is not None:
            require(isinstance(expected, str) and re.fullmatch(r"[0-9a-fA-F]{64}", expected),
                    f"invalid SHA256 for {path}")
        digest = sha(path)
        require(expected is None or digest == expected.lower(), f"hash mismatch: {path}")
        previous = self.hashes.get(str(path))
        require(previous is None or previous == digest, f"input changed during scoring: {path}")
        self.hashes[str(path)] = digest
        return path

    def frozen(self, path):
        path = Path(path)
        require(str(path) in self.files, f"required input absent from verifier freeze: {path}")
        self.file(path, self.files[str(path)])
        self.checked_frozen.add(str(path))
        return path

    def json(self, path, frozen=False):
        path = self.frozen(path) if frozen else self.file(path)
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json,
                           parse_constant=_bad_constant)
        finite_tree(value, str(path))
        return value

    def recheck(self):
        for path, expected in self.hashes.items():
            require(sha(path) == expected, f"input changed before publication: {path}")


def key(row):
    return row["scene_id"], row["seed_index"]


def unique_rows(rows, where):
    require(isinstance(rows, list), f"{where}: rows must be a list")
    result = {}
    for row in rows:
        require(isinstance(row, dict), f"{where}: invalid row")
        sid, si = key(row)
        require(isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9_]+", sid),
                f"{where}: invalid scene ID")
        require(type(si) is int and si in (0, 1), f"{where}: invalid seed index")
        require((sid, si) not in result, f"{where}: duplicate unit {(sid, si)}")
        result[sid, si] = row
    return result


def full_main_keys():
    return {(f"{family}_{variant}", si) for family in verifier.FAMILIES
            for variant in VARIANTS for si in (0, 1)}


def exact_keys(actual, expected, where):
    actual, expected = set(actual), set(expected)
    require(actual == expected,
            f"{where}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def validate_packet(packet, update, contract, labels, freeze_sha256):
    require(packet.get("candidate") == CANDIDATE and packet.get("update") == update, "wrong packet")
    require(packet.get("verifier_sha256") == freeze_sha256, "packet uses another verifier freeze")
    require(packet.get("contains_target_geometry_or_masks") is False, "packet target-input flag")
    rows = packet["rows"]
    indexed = unique_rows(rows, "packet")
    groups = {kind: [] for kind in ("main", "legacy", "challenge")}
    for row in rows:
        sid, si = key(row)
        kind = row["kind"]
        require(kind in groups, f"unexpected packet kind: {kind}")
        require(type(row["noise_seed"]) is int, f"{sid}: invalid noise seed")
        if kind == "legacy":
            require(sid in contract["legacy"] and row["variant"] == "original" and si == 0,
                    f"unapproved legacy unit: {(sid, si)}")
        else:
            require(sid in labels, f"missing private label entry: {sid}")
            record = labels[sid]["record"]
            require(record["split"] == "SUPPLEMENTARY_DEV", f"wrong label split: {sid}")
            require(record["scene_id"] == sid and record["family_id"] == row["family"]
                    and record["variant"] == row["variant"], f"packet/label identity mismatch: {sid}")
            require(sid == f"{row['family']}_{row['variant']}", f"noncanonical scene: {sid}")
            require(record["map_sha256"] == row["map_sha256"]
                    and record["seeds"]["structure"][si] == row["noise_seed"],
                    f"packet/label map or seed mismatch: {sid}")
            require((kind == "main" and row["variant"] in VARIANTS)
                    or (kind == "challenge" and row["variant"] == "challenge1p5m"),
                    f"variant in wrong cohort: {sid}")
        groups[kind].append(row)
    exact_keys(map(key, groups["legacy"]), {(s, 0) for s in contract["legacy"]}, "packet legacy")
    if update >= 100:
        exact_keys(map(key, groups["main"]), full_main_keys(), "packet main")
    else:
        # make_eval_packets uses the old fixed +100 panel for the new +20 screen.
        expected = {(f"courtyard_new_{n:03d}_{v}", 0)
                    for n in (1, 2) for v in ("closed", "open3m")}
        exact_keys(map(key, groups["main"]), expected, "packet +20 diagnostic")
    expected_challenge = {(f"{f}_challenge1p5m", 0) for f in verifier.FAMILIES} if update == 300 else set()
    exact_keys(map(key, groups["challenge"]), expected_challenge, "packet challenge")
    return indexed, groups


def binary(value, shape, where):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind in "biuf", f"invalid label array: {where}")
    require(np.isfinite(value).all() and ((value == 0) | (value == 1)).all(), f"nonbinary label: {where}")
    return value.astype(bool, copy=True)


def load_labels(inputs, manifest, scenes):
    result = {}
    for sid in sorted(scenes):
        entry = manifest[sid]
        path = Path(entry["path"])
        require(path == JOB / "verifier_private" / f"{sid}.npz", f"unexpected private label path: {path}")
        inputs.frozen(path)
        require(inputs.hashes[str(path)] == entry["sha256"], f"label manifest hash mismatch: {sid}")
        names = ("target",) + MASKS
        if entry["record"]["variant"] != "closed":
            names += ("shared_closed_walls",)
        with np.load(path, allow_pickle=False) as arrays:
            result[sid] = {name: binary(arrays[name], (64, 64) if name == "eroded_xy" else (64, 64, 64),
                                        f"{sid}/{name}") for name in names}
    return result


def support(coords, where):
    require(isinstance(coords, torch.Tensor) and coords.device.type == "cpu", f"{where}: CPU Q tensor required")
    array = coords.detach().numpy()
    result = verifier.dense(array)
    require(len(array) == int(result.sum()), f"{where}: duplicate Q coordinates")
    return result


def load_raw(inputs, path, expected, generation=False):
    inputs.file(path, expected)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(raw, dict), f"{path}: raw dictionary required")
    prediction = support(raw["predicted_Q"], str(path))
    for name in ("predicted_S", "structure_noise"):
        require(not generation or name in raw, f"{path}: missing {name}")
        if name in raw:
            value = raw[name]
            require(isinstance(value, torch.Tensor) and value.is_floating_point()
                    and tuple(value.shape) == (1, 8, 16, 16, 16)
                    and bool(torch.isfinite(value).all()), f"{path}: invalid/nonfinite {name}")
    return prediction, raw


def load_legacy(inputs, contract):
    audit = inputs.json(JOB / "audit/LEGACY_RAW_REGRESSION.json")
    require(audit.get("PASS") is True and audit.get("units") == 28, "legacy raw regression incomplete")
    indexed = {}
    for row in audit["rows"]:
        identity = row["arm"], row["scene"]
        require(identity not in indexed, f"duplicate legacy audit row: {identity}")
        number(row["value"], f"legacy audit {identity}")
        indexed[identity] = row
    exact_keys(indexed, {(arm, sid) for arm in HISTORICAL_ARMS for sid in contract["legacy"]}, "legacy audit")
    targets, references = {}, {}
    for sid in contract["legacy"]:
        row = indexed["MIX0", sid]
        target_path = NATIVE / "paired_scene_features_v1" / sid / "structure_native.pt"
        require(row["target_path"] == str(target_path), f"unapproved legacy target: {sid}")
        for arm in HISTORICAL_ARMS:
            other = indexed[arm, sid]
            require(other["target_path"] == str(target_path)
                    and other["target_sha256"] == row["target_sha256"], f"conflicting legacy target: {sid}")
        inputs.file(target_path, row["target_sha256"])
        target = torch.load(target_path, map_location="cpu", weights_only=True)
        targets[sid] = support(target["Q0"], str(target_path))
        require(bool((targets[sid] & verifier.NG).any()), f"empty legacy target support: {sid}")
        references[sid] = row
    return targets, references


def legacy_score(prediction, target):
    return {"endpoint": "original", "predicted_Q_cells": int(prediction.sum()),
            "target_Q_cells": int(target.sum()),
            "non_ground_native_IoU": verifier.iou(prediction, target, verifier.NG),
            "non_ground_precision": verifier.fraction(int((prediction & target & verifier.NG).sum()),
                                                       int((prediction & verifier.NG).sum())),
            "non_ground_recall": verifier.recall(prediction, target & verifier.NG),
            "non_ground_support_ratio": verifier.fraction(int((prediction & verifier.NG).sum()),
                                                           int((target & verifier.NG).sum()))}


def exact_mean(rows, field, where):
    require(bool(rows), f"{where}: empty mean for {field}")
    values = [number(row[field], f"{where}/{field}") for row in rows]
    return float(np.mean(values))


def aggregate(main_rows, legacy_rows, expected_units, legacy_scenes):
    indexed = unique_rows(main_rows, "aggregation main")
    exact_keys(indexed, expected_units, "aggregation main")
    exact_keys(unique_rows(legacy_rows, "aggregation legacy"), {(s, 0) for s in legacy_scenes}, "aggregation legacy")
    families = []
    for family in sorted({r["family_id"] for r in main_rows}):
        rows = [r for r in main_rows if r["family_id"] == family]
        opened = [r for r in rows if r["variant"] in ("open3m", "open4m")]
        closed = [r for r in rows if r["variant"] == "closed"]
        item = {"family_id": family, **{f: exact_mean(opened, f, family) for f in FIELDS}}
        item.update(courtyard_error=(item[FIELDS[0]] + item[FIELDS[1]]) / 2,
                    closed_closure_recall=exact_mean(closed, "closed_closure_recall", family),
                    open_units=len(opened), closed_units=len(closed),
                    opening_variants=sorted({r["variant"] for r in opened}),
                    generation_seed_indices=sorted({r["seed_index"] for r in rows}))
        families.append(item)
    summary = {f: exact_mean(families, f, "family-weighted summary")
               for f in FIELDS + ("courtyard_error", "closed_closure_recall")}
    summary["legacy_non_ground_IoU"] = exact_mean(legacy_rows, "non_ground_native_IoU", "four original legacy controls")
    return families, summary


def diagnostic_means(rows, fields):
    """No nanmean/filtering: one undefined unit makes that pooled mean undefined."""
    result = {}
    for field in fields:
        missing = [i for i, row in enumerate(rows) if row.get(field) is None]
        values = [number(row[field], field) for row in rows if row.get(field) is not None]
        result[field] = {"mean": float(np.mean(values)) if rows and not missing else None,
                         "total_units": len(rows), "defined_units": len(values),
                         "undefined_row_indices": missing}
    return result


def baseline(inputs, labels, legacy_targets, legacy_refs, legacy_scenes):
    historical = inputs.json(JOB / "audit/HISTORICAL_ROWS.json")["MIX0"]
    selected = [r for r in historical if r["milestone"] == 300]
    indexed = unique_rows(selected, "MIX0 +300")
    exact_keys(indexed, full_main_keys(), "MIX0 +300")
    hashes = inputs.json(JOB / "verifier_private/INPUT_HASHES.json", frozen=True)
    rows = []
    for (sid, si), old in sorted(indexed.items()):
        path = DATA / "samples/R0_plus300" / sid / f"seed{si}" / "RAW_STRUCTURE.pt"
        require(old["raw_path"] == str(path) and old["raw_sha256"] == hashes[str(path)], f"MIX0 raw provenance: {sid}")
        prediction, _ = load_raw(inputs, path, hashes[str(path)])
        label = labels[sid]
        variant = sid.rsplit("_", 1)[1]
        value = verifier.score(prediction, label["target"], label, variant)
        for field in FIELDS + ("closed_closure_recall",):
            if old[field] is not None:
                require(abs(number(value[field], field) - number(old[field], field)) <= 1e-10,
                        f"MIX0 regression mismatch: {sid}/seed{si}/{field}")
        rows.append({**value, "scene_id": sid, "seed_index": si, "variant": variant,
                     "family_id": sid.rsplit("_", 1)[0]})
    legacy_rows = []
    for sid in legacy_scenes:
        ref = legacy_refs[sid]
        path = DATA / "legacy/R0" / sid / "original/RAW_STRUCTURE.pt"
        require(ref["raw_path"] == str(path), f"unexpected MIX0 legacy path: {sid}")
        prediction, _ = load_raw(inputs, path, ref["raw_sha256"])
        value = legacy_score(prediction, legacy_targets[sid])
        require(abs(number(value["non_ground_native_IoU"], sid) - ref["value"]) <= 1e-10,
                f"MIX0 legacy regression mismatch: {sid}")
        legacy_rows.append({**value, "scene_id": sid, "seed_index": 0})
    families, summary = aggregate(rows, legacy_rows, full_main_keys(), legacy_scenes)
    for field, threshold in verifier.THRESHOLDS.items():
        require(abs(summary[field] - .02 - threshold) <= 1e-12, f"baseline does not reproduce frozen threshold: {field}")
    return rows, legacy_rows, families, summary


def discover_outputs(root, required_keys):
    require(root.is_dir(), f"missing evaluation directory: {root}")
    found = {}
    for path in root.rglob("RAW_STRUCTURE.pt"):
        relative = path.relative_to(root)
        require(len(relative.parts) == 3 and re.fullmatch(r"seed[01]", relative.parts[1]),
                f"unexpected raw-output layout: {path}")
        require(path.resolve().is_relative_to(root.resolve()), f"raw output escapes evaluation directory: {path}")
        identity = relative.parts[0], int(relative.parts[1][-1])
        require(identity not in found, f"duplicate output unit: {identity}")
        found[identity] = path
    exact_keys(found, required_keys, f"output coverage {root}")
    return found


def score_seed(inputs, seed, update, packet_index, labels, legacy_targets, base_rows,
               base_families, base_summary, legacy_scenes):
    root = JOB / "evaluation" / CANDIDATE / f"seed{seed}" / f"plus{update:03d}"
    paths = discover_outputs(root, packet_index)
    receipt = inputs.json(root / "COMPLETE.json")
    meta_path = JOB / "runs" / CANDIDATE / f"seed{seed}" / f"CHECKPOINT_{update:06d}.json"
    checkpoint = inputs.json(meta_path)
    require(checkpoint.get("candidate") == CANDIDATE and checkpoint.get("seed") == seed
            and checkpoint.get("update") == update, "wrong checkpoint identity")
    checkpoint_path = meta_path.parent / f"checkpoint_{update:06d}.pt"
    require(checkpoint["path"] == str(checkpoint_path), "noncanonical checkpoint path")
    inputs.file(checkpoint_path, checkpoint["sha256"])  # Hash only; never deserialize model weights.
    require(receipt["cases"] == len(packet_index)
            and receipt["checkpoint_sha256"] == checkpoint["sha256"], "incomplete/wrong root receipt")
    require(receipt["branch_calls"]["mapped_nonzero"] > 0
            and receipt["branch_calls"]["unknown_exact_zero"] > 0, "missing inference branch checks")
    groups = {kind: [] for kind in ("main", "legacy", "challenge")}
    predictions, noises, roofs = {}, {}, []
    for identity, unit in sorted(packet_index.items()):
        sid, si = identity
        path = paths[identity]
        complete = inputs.json(path.parent / "COMPLETE.json")
        require(complete["packet"] == unit, f"unit receipt differs from required packet: {identity}")
        prediction, raw = load_raw(inputs, path, complete["files"]["RAW_STRUCTURE.pt"], generation=True)
        require(raw.get("no_target_input") is True and raw.get("checkpoint") == checkpoint,
                f"wrong generation provenance: {identity}")
        if unit["kind"] == "legacy":
            value = legacy_score(prediction, legacy_targets[sid])
        else:
            label = labels[sid]
            value = verifier.score(prediction, label["target"], label, unit["variant"])
            predictions[identity], noises[identity] = prediction, raw["structure_noise"]
            if unit["kind"] == "main" and unit["variant"] != "closed":
                require(not value["empty_eroded_core_not_scored_success"], f"undefined robust opening core: {identity}")
            if unit["variant"] == "closed":
                roofs.append({"scene_id": sid, "seed_index": si,
                              **verifier.roof_diagnostic(prediction, label["closed_closure"])})
        finite_tree(value, str(identity))
        groups[unit["kind"]].append({**value, "scene_id": sid, "seed_index": si,
            "family_id": unit["family"], "variant": unit["variant"], "kind": unit["kind"],
            "raw_path": str(path), "raw_sha256": inputs.hashes[str(path)]})
    pairs = {"main": [], "challenge": []}
    for identity, prediction in sorted(predictions.items()):
        unit = packet_index[identity]
        if unit["variant"] == "closed":
            continue
        closed_key = f"{unit['family']}_closed", unit["seed_index"]
        require(closed_key in predictions, f"missing closed counterpart: {identity}")
        closed_label, opened_label = labels[closed_key[0]], labels[identity[0]]
        noise_equal = (noises[closed_key].dtype == noises[identity].dtype
                       and torch.equal(noises[closed_key], noises[identity]))
        value = verifier.paired(predictions[closed_key], prediction, closed_label["target"],
                                opened_label["target"], opened_label["shared_closed_walls"],
                                closed_label["protected"] & opened_label["protected"],
                                opened_label["free_air"] | opened_label["overhead"], noise_equal)
        pairs[unit["kind"]].append({"scene_id": identity[0], "seed_index": identity[1],
                                   "closed_scene_id": closed_key[0], **value})
    expected_main = {k for k, r in packet_index.items() if r["kind"] == "main"}
    families, summary = aggregate(groups["main"], groups["legacy"], expected_main, legacy_scenes)
    gains = []
    for si in sorted({r["seed_index"] for r in groups["main"]}):
        candidate = [r for r in groups["main"] if r["variant"] != "closed" and r["seed_index"] == si]
        reference = [r for r in base_rows if key(r) in expected_main and r["variant"] != "closed" and r["seed_index"] == si]
        exact_keys(map(key, candidate), map(key, reference), "generation-seed baseline selection")
        errors = [exact_mean([{ "error": (r[FIELDS[0]] + r[FIELDS[1]]) / 2} for r in rows],
                             "error", f"generation seed {si}") for rows in (reference, candidate)]
        gains.append(errors[0] - errors[1])
    gate = verifier.native_gate(summary, base_summary, families,
                                {r["family_id"]: r for r in base_families}, gains) if update >= 100 else None
    extra_fields = ("non_ground_precision", "non_ground_recall", "non_ground_support_ratio",
                    "map_horizontal_support_IoU", "unknown_reference_IoU")
    result = {"training_seed": seed, "checkpoint": checkpoint, "unit_counts": {k: len(v) for k, v in groups.items()},
              "main_scores": groups["main"], "legacy_scores": groups["legacy"], "challenge_scores": groups["challenge"],
              "per_family": families, "summary": summary, "generation_seed_gains": gains,
              "native_Q_gate": gate, "paired_edits": pairs, "closed_roof_diagnostics": roofs,
              "diagnostic_means": {"main_support": diagnostic_means(groups["main"], extra_fields),
                  "closed_roof": diagnostic_means(roofs, ROOF_FIELDS),
                  "main_pairs": diagnostic_means(pairs["main"], PAIR_FIELDS),
                  "challenge": diagnostic_means(groups["challenge"], FIELDS + extra_fields),
                  "challenge_pairs": diagnostic_means(pairs["challenge"], PAIR_FIELDS)}}
    finite_tree(result, f"seed{seed}")
    return result


def main(argv=None):
    global CANDIDATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["g1_a", "g2_a"], default="g1_a")
    parser.add_argument("--update", type=int, choices=[20, 100, 300], required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Explicit single-training-seed report; default requires both")
    args = parser.parse_args(argv)
    CANDIDATE = args.candidate
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    suffix = f"_seed{args.seed}" if args.seed is not None else ""
    output = JOB / "scores" / f"{CANDIDATE}{suffix}_plus{args.update:03d}.json"
    require(not output.exists(), f"exclusive report already exists: {output}")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    inputs = Inputs()
    contract = inputs.json(JOB / "CONTRACT.json", frozen=True)
    require(contract["thresholds"] == verifier.THRESHOLDS, "frozen threshold mismatch")
    require(contract["main_families"] == list(verifier.FAMILIES)
            and contract["main_variants"] == list(VARIANTS)
            and contract["generation_seed_indices"] == [0, 1], "unsupported frozen main selection")
    require(len(contract["legacy"]) == len(set(contract["legacy"])) == 4
            and set(contract["legacy"]) == {f"graph_{n:04d}_a0" for n in range(48, 52)},
            "legacy selection must be the four approved original development controls")
    manifest = inputs.json(JOB / "verifier_private/LABEL_MANIFEST.json", frozen=True)
    packet_path = JOB / "packets" / f"{CANDIDATE}_{args.update}.json"
    packet = inputs.json(packet_path)
    indexed, packet_groups = validate_packet(packet, args.update, contract, manifest, inputs.freeze_sha256)
    # Check both arms' output coverage before loading private numerical labels.
    for seed in seeds:
        discover_outputs(JOB / "evaluation" / CANDIDATE / f"seed{seed}" / f"plus{args.update:03d}", indexed)
    scenes = {sid for sid, _ in full_main_keys()} | {r["scene_id"] for r in packet_groups["challenge"]}
    labels = load_labels(inputs, manifest, scenes)
    legacy_targets, legacy_refs = load_legacy(inputs, contract)
    base_rows, base_legacy, base_families, base_summary = baseline(inputs, labels, legacy_targets, legacy_refs, contract["legacy"])
    selected = {key(r) for r in packet_groups["main"]}
    baseline_subset_families, baseline_subset_summary = aggregate(
        [r for r in base_rows if key(r) in selected], base_legacy, selected, contract["legacy"])
    results = {f"seed{seed}": score_seed(inputs, seed, args.update, indexed, labels, legacy_targets,
                base_rows, base_families, base_summary, contract["legacy"]) for seed in seeds}
    report = {"schema_version": 1, "status": "SCORED_ONLY", "candidate": CANDIDATE, "update": args.update,
        "training_seeds": seeds, "partial_diagnostic_only": args.update == 20,
        "required_packet_units": {k: len(v) for k, v in packet_groups.items()},
        "thresholds": dict(verifier.THRESHOLDS),
        "baseline": {"arm": "MIX0", "milestone": 300, "summary": base_summary, "per_family": base_families,
                     "packet_subset_summary": baseline_subset_summary, "packet_subset_per_family": baseline_subset_families},
        "results": results, "admission_evaluated": False, "final_acceptance_evaluated": False,
        "HUMAN_REVIEW_PENDING": True, "native_gate_is_not_admission_or_final_acceptance": True,
        "challenge_excluded_from_main_gate": True, "tolerance_does_not_relax_exact_gate": True,
        "original_sealed_test_read": False, "reserved_heldout_read": False,
        "mean_policy": "Open fields: equal units within family, then equal families; closure: closed units only; legacy: four original endpoints. Undefined diagnostics are null, never dropped.",
        "provenance": {"driver_sha256": sha(Path(__file__)), "verifier_freeze_sha256": inputs.freeze_sha256,
                       "frozen_files_checked": sorted(inputs.checked_frozen), "input_sha256": dict(inputs.hashes)}}
    finite_tree(report)
    json.dumps(report, allow_nan=False)  # Validate before save can create a file.
    inputs.recheck()
    save(output, report)
    print(json.dumps({"report": str(output), "training_seeds": seeds, "update": args.update, "status": "SCORED_ONLY"}))


if __name__ == "__main__":
    main()
