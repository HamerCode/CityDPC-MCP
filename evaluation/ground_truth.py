"""Regenerate data/ground_truth.json from the evaluation dataset with CityDPC.

    python evaluation/ground_truth.py
"""

import json
from pathlib import Path

from cases import RAISE_BUILDING_SPEC, TEST_BUILDING_SPEC, TEST_CASES

ROOT = Path(__file__).resolve().parent


def generate_ground_truth(dataset_path: Path) -> dict:
    """
    Generiert Ground Truth für alle Test Cases mit CityDPC.

    Args:
        dataset_path: Pfad zum CityJSON Dataset

    Returns:
        dict: Ground Truth Daten mit Solutions für alle Test Cases
    """
    try:
        from citydpc import Dataset
        from citydpc.core import input as citydpc_input
        from citydpc.core.input.cityjsonInput import load_buildings_from_json_file
        from citydpc.tools import partywall
    except ImportError:
        raise ImportError("CityDPC not installed. Run `uv sync --extra eval` in the repository root.")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load Dataset
    dataset = Dataset()
    citydpc_input.set_roof_volume_calculation(True)
    load_buildings_from_json_file(dataset, str(dataset_path))

    # Helper Functions
    # Generate Solutions
    case_results = {}

    # 1. List Buildings
    ids = sorted(dataset.buildings.keys())
    case_results["list_buildings"] = {"building_ids": ids, "total": len(ids)}

    # 2. Highest Measured Height
    highest_id = None
    highest_value = None
    missing = 0
    for building in dataset.get_building_list():
        height = building.measuredHeight
        if height is None:
            missing += 1
            continue
        if highest_value is None or height > highest_value:
            highest_value = height
            highest_id = building.gml_id
    case_results["highest_measured_height"] = {
        "building_id": highest_id,
        "measuredHeight": highest_value,
        "missing_count": missing,
    }

    # 3. Count Party Walls
    walls = partywall.get_party_walls(dataset)
    case_results["count_party_walls"] = {
        "party_wall_count": len(walls),
        "distinct_buildings_involved": len({str(w[i]) for w in walls for i in (0, 2)}),
        "total_area_m2": round(sum(float(w[4]) for w in walls), 3),
    }

    # 4. Raise Building By ID - Verwendet RAISE_BUILDING_SPEC
    # Verifiziere, dass die Spec mit dem Dataset übereinstimmt
    spec_building_id = RAISE_BUILDING_SPEC["target_building_id"]
    spec_old_height = RAISE_BUILDING_SPEC["old_height"]
    spec_new_height = RAISE_BUILDING_SPEC["new_height"]
    
    # Prüfe ob das spezifizierte Gebäude existiert und die Höhe stimmt.
    target_building = dataset.buildings.get(spec_building_id)
    
    if target_building is None:
        print(f"[WARNUNG] RAISE_BUILDING_SPEC target_building_id '{spec_building_id}' existiert nicht")
        actual_height = None
    else:
        actual_height = target_building.measuredHeight
    
    if actual_height and abs(spec_old_height - actual_height) > 0.01:
        print(f"[WARNUNG] RAISE_BUILDING_SPEC old_height {spec_old_height} != "
              f"tatsaechliche Hoehe {actual_height}")
    
    case_results["raise_building_by_id"] = {
        "building_id": spec_building_id,
        "old_measuredHeight": spec_old_height,
        "new_measuredHeight": spec_new_height,
        "increase_by": RAISE_BUILDING_SPEC["increase_by"],
        "modification_performed": True
    }

    # 5. Add Building - Verwendet TEST_BUILDING_SPEC
    new_id = TEST_BUILDING_SPEC["id"]
    footprint = TEST_BUILDING_SPEC["groundCoordinates"]
    ground_height = TEST_BUILDING_SPEC["groundSurfaceHeight"]
    geometry_height = TEST_BUILDING_SPEC["measuredHeight"]
    roof_type = TEST_BUILDING_SPEC["roofType"]
    roof_height = TEST_BUILDING_SPEC.get("roofHeight", 3.0)
    
    # Berechne geographicalExtent
    xs = [c[0] for c in footprint[:-1]]  # Ohne den schließenden Punkt
    ys = [c[1] for c in footprint[:-1]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z = ground_height
    max_z = ground_height + geometry_height
    
    # Erstelle das erwartete CityJSON-Objekt
    expected_cityjson_object = {
        "type": "Building",
        "attributes": {
            "measuredHeight": TEST_BUILDING_SPEC["measuredHeight"],
            "storeysAboveGround": TEST_BUILDING_SPEC["storeysAboveGround"],
            "roofType": TEST_BUILDING_SPEC["roofType"],
            "function": TEST_BUILDING_SPEC["function"],
        },
        "geographicalExtent": [min_x, min_y, min_z, max_x, max_y, max_z],
        # geometry wird vom LLM erstellt - wir prüfen nur Struktur
    }
    
    case_results["add_building"] = {
        "building_id": new_id,
        "expected_spec": TEST_BUILDING_SPEC,
        "cityjson_object": expected_cityjson_object
    }

    # Build Solutions Dict
    solutions = {}
    for test_case in TEST_CASES:
        case_id = test_case["id"]
        solutions[case_id] = {
            "category": test_case["category"],
            "prompt": test_case["prompt"],
            "output": case_results.get(case_id),
        }

    output = {
        "dataset": dataset_path.name,
        "model": "citydpc-local",
        "test_building_spec": TEST_BUILDING_SPEC,
        "raise_building_spec": RAISE_BUILDING_SPEC,
        "solutions": solutions,
    }

    return output


if __name__ == "__main__":
    dataset_path = ROOT / "data" / "evaluation.city.json"
    output_path = ROOT / "data" / "ground_truth.json"
    
    print(f"Generiere Ground Truth aus {dataset_path}...")
    
    # Ground Truth generieren
    result = generate_ground_truth(dataset_path)
    
    # Als JSON speichern
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Ground Truth erfolgreich gespeichert in {output_path}")
    print(f"  - Gebaeude gesamt: {result['solutions']['list_buildings']['output']['total']}")
    print(f"  - Hoechstes Gebaeude: {result['solutions']['highest_measured_height']['output']['building_id']} ({result['solutions']['highest_measured_height']['output']['measuredHeight']}m)")
    print(f"  - Party Walls: {result['solutions']['count_party_walls']['output']['party_wall_count']}")
    print(f"  - Test-Gebäude ID: {result['test_building_spec']['id']}")
    print(f"  - Raise-Building ID: {result['raise_building_spec']['target_building_id']}")
