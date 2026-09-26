"""
Validierungs-Logic für Test Results gegen Ground Truth.

Unterstützt sowohl JSON-Schema-basierte Validierung (Query/Analytic)
als auch Dataset-Modifikations-Validierung (State-Change Tests).

Erweiterte Validierung für State-Change Tests:
- CityJSON Struktur-Validierung (parseable, required fields)
- Geometry-Validierung (Vertex-Referenzen, Boundaries)
- CityDPC Import-Test (Dataset ladbar durch Library)
"""

import json
import re
import traceback
from pathlib import Path


def extract_json_from_response(response_text: str) -> dict | None:
    """Extrahiert JSON aus response_text."""
    response_text = re.sub(r"\[THINK\].*?\[/THINK\]", "", response_text, flags=re.DOTALL).strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: Versuche JSON Block zu finden
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def extract_city_object_from_text(text: str) -> dict | None:
    """
    Extrahiert ein CityObject (oder mehrere) aus Freitext.
    
    Sucht nach JSON-Blöcken, die CityJSON-Strukturen enthalten.
    Gibt das erste gefundene CityObject zurück.
    """
    # Versuche verschiedene Muster
    patterns = [
        # JSON-Codeblock
        r'```json\s*(\{[\s\S]*?\})\s*```',
        # JSON-Codeblock ohne Sprache
        r'```\s*(\{[\s\S]*?\})\s*```',
        # Rohes JSON (mit Building-ID als Key)
        r'(\{\s*"[A-Za-z0-9_-]+"\s*:\s*\{[\s\S]*?"type"\s*:\s*"Building"[\s\S]*?\}\s*\})',
        # Inneres CityObject direkt
        r'(\{\s*"type"\s*:\s*"Building"[\s\S]*?\})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for match in matches:
            try:
                parsed = json.loads(match)
                # Prüfe ob es ein CityObject oder ein Wrapper ist
                if isinstance(parsed, dict):
                    # Direkt ein CityObject mit "type": "Building"
                    if parsed.get("type") == "Building":
                        return {"extracted_object": parsed, "has_id_wrapper": False}
                    # Wrapper mit ID als Key: {"BUILDING_ID": {...}}
                    for key, value in parsed.items():
                        if isinstance(value, dict) and value.get("type") == "Building":
                            return {"building_id": key, "extracted_object": value, "has_id_wrapper": True}
            except json.JSONDecodeError:
                continue
    
    return None


def compare_building_list(extracted: dict, expected: dict) -> dict:
    """Vergleicht Building List Ergebnisse."""
    extracted_ids = set(extracted.get("building_ids", []))
    expected_ids = set(expected.get("output", {}).get("building_ids", []))

    extracted_total = extracted.get("total", 0)
    expected_total = expected.get("output", {}).get("total", 0)

    ids_match = extracted_ids == expected_ids
    total_match = extracted_total == expected_total

    missing = expected_ids - extracted_ids
    extra = extracted_ids - expected_ids

    return {
        "ids_match": ids_match,
        "total_match": total_match,
        "accuracy": 1.0 if (ids_match and total_match) else 0.0,
        "precision": len(extracted_ids & expected_ids) / len(extracted_ids) if extracted_ids else 0.0,
        "recall": len(extracted_ids & expected_ids) / len(expected_ids) if expected_ids else 0.0,
        "missing_ids": list(missing),
        "extra_ids": list(extra),
        "missing_count": len(missing),
        "extra_count": len(extra),
    }


def compare_highest_building(extracted: dict, expected: dict) -> dict:
    """Vergleicht Highest Building Ergebnisse."""
    extracted_id = extracted.get("building_id", "")
    expected_id = expected.get("output", {}).get("building_id", "")

    extracted_height = extracted.get("measuredHeight")
    expected_height = expected.get("output", {}).get("measuredHeight")

    id_match = extracted_id == expected_id

    # Toleranz für Floating Point: 0.001m = 1mm
    height_match = False
    if extracted_height is not None and expected_height is not None:
        height_match = abs(extracted_height - expected_height) < 0.001

    accuracy = 1.0 if (id_match and height_match) else 0.5 if id_match else 0.0

    return {
        "id_match": id_match,
        "height_match": height_match,
        "accuracy": accuracy,
        "extracted_id": extracted_id,
        "expected_id": expected_id,
        "extracted_height": extracted_height,
        "expected_height": expected_height,
        "height_difference": abs(extracted_height - expected_height) if (extracted_height and expected_height) else None,
    }


def compare_party_walls(extracted: dict, expected: dict) -> dict:
    """Vergleicht Party-Wall-Count Ergebnisse (exakter Match)."""
    if isinstance(extracted, int):
        extracted = {"party_wall_count": extracted}

    extracted_count = extracted.get("party_wall_count")
    expected_count = expected.get("output", {}).get("party_wall_count")

    count_match = (
        extracted_count is not None
        and expected_count is not None
        and int(extracted_count) == int(expected_count)
    )
    accuracy = 1.0 if count_match else 0.0

    return {
        "count_match": count_match,
        "accuracy": accuracy,
        "extracted_count": extracted_count,
        "expected_count": expected_count,
        "count_difference": (
            abs(int(extracted_count) - int(expected_count))
            if (extracted_count is not None and expected_count is not None)
            else None
        ),
    }


def compare_building_modification(extracted: dict, expected: dict) -> dict:
    """Vergleicht Building Modification Ergebnisse."""
    extracted_id = extracted.get("building_id", "")
    expected_id = expected.get("output", {}).get("building_id", "")

    extracted_height = extracted.get("new_measuredHeight")
    expected_height = expected.get("output", {}).get("new_measuredHeight")

    id_match = extracted_id == expected_id

    # Toleranz: 0.001m
    height_match = False
    if extracted_height is not None and expected_height is not None:
        height_match = abs(extracted_height - expected_height) < 0.001

    accuracy = 1.0 if (id_match and height_match) else 0.5 if id_match else 0.0

    return {
        "id_match": id_match,
        "height_match": height_match,
        "accuracy": accuracy,
        "extracted_id": extracted_id,
        "expected_id": expected_id,
        "extracted_height": extracted_height,
        "expected_height": expected_height,
    }


def compare_new_building(extracted: dict, expected: dict) -> dict:
    """Vergleicht New Building Ergebnisse - strukturelle Vollständigkeit."""
    has_id = bool(extracted.get("building_id", "").strip())
    has_height = extracted.get("measuredHeight") is not None
    has_storeys = extracted.get("storeysAboveGround") is not None
    has_roof_type = bool(extracted.get("roofType", "").strip())
    has_function = bool(extracted.get("function", "").strip())

    # Vollständigkeitsprüfung
    required_fields = [has_id, has_height, has_storeys, has_roof_type, has_function]
    completeness = sum(required_fields) / len(required_fields)
    accuracy = completeness

    return {
        "has_building_id": has_id,
        "has_measuredHeight": has_height,
        "has_storeysAboveGround": has_storeys,
        "has_roofType": has_roof_type,
        "has_function": has_function,
        "completeness": completeness,
        "accuracy": accuracy,
        "extracted_id": extracted.get("building_id", ""),
    }


def validate_test_result(
    test_case_id: str,
    response_text: str,
    ground_truth_solutions: dict,
) -> dict:
    """
    Validiert Test-Ergebnis basierend auf JSON Response und Ground Truth.

    Args:
        test_case_id: ID des Test Cases
        response_text: Response Text vom AI Model
        ground_truth_solutions: Ground Truth Daten

    Returns:
        dict: Validierungs-Ergebnis mit accuracy, extracted_data, etc.
    """
    # 1. Extrahiere JSON aus Response
    extracted = extract_json_from_response(response_text)

    if extracted is None:
        return {
            "validated": False,
            "json_parsed": False,
            "accuracy": 0.0,
            "reason": "Could not parse JSON from response"
        }

    if not isinstance(extracted, dict) and test_case_id != "count_party_walls":
        return {
            "validated": False,
            "json_parsed": True,
            "accuracy": 0.0,
            "reason": f"Parsed JSON is {type(extracted).__name__}, expected object",
            "extracted_data": extracted,
        }

    # 2. Hole Ground Truth für diesen Test Case
    expected = ground_truth_solutions.get(test_case_id, {})

    if not expected:
        return {
            "validated": False,
            "json_parsed": True,
            "accuracy": None,
            "reason": f"No ground truth for test case {test_case_id}",
            "extracted_data": extracted
        }

    # 3. Vergleiche basierend auf Test Case
    comparison_funcs = {
        "list_buildings": compare_building_list,
        "highest_measured_height": compare_highest_building,
        "count_party_walls": compare_party_walls,
        "raise_building_by_id": compare_building_modification,
        "add_building": compare_new_building,
    }

    compare_func = comparison_funcs.get(test_case_id)
    if not compare_func:
        return {
            "validated": False,
            "json_parsed": True,
            "accuracy": 0.0,
            "reason": f"No comparison function for {test_case_id}",
            "extracted_data": extracted
        }

    result = compare_func(extracted, expected)
    result["validated"] = True
    result["json_parsed"] = True
    result["extracted_data"] = extracted

    return result


# ============================================================================
# DATASET MODIFICATION VALIDATION (für State-Change Tests)
# ============================================================================

def validate_dataset_modification(
    test_id: str,
    setup_key: str,
    output_text: str,
    dataset_path: Path,
    expected_building_spec: dict | None = None,
    expected_raise_spec: dict | None = None
) -> dict:
    """
    Validiert ob eine Dataset-Modifikation erfolgreich war.
    
    Strategie:
    - MCP: Dataset wurde direkt modifiziert -> Datei prüfen
    - Code-Interpreter: Dataset könnte modifiziert sein ODER Output enthält CityObject
    - No-Tools: Nur Output-Extraktion möglich (keine echte Modifikation)
    - patched_no_tools: Patch wurde lokal angewendet -> Datei prüfen
    - ci_fallback: Output wurde lokal angewendet -> Datei prüfen
    
    Args:
        test_id: "raise_building_by_id" oder "add_building"
        setup_key: "mcp", "code_interpreter", oder "no_tools"
        output_text: Response Text vom Model
        dataset_path: Pfad zum Dataset
        expected_building_spec: Erwartete Gebäudespezifikation (für add_building)
        expected_raise_spec: Erwartete Modifikation (für raise_building_by_id)
    
    Returns:
        dict: Validierungsergebnis mit accuracy, modification_verified, details
    """
    result = {
        "validated": False,
        "modification_verified": False,
        "accuracy": 0.0,
        "setup_key": setup_key,
        "test_id": test_id,
        "checks": {}
    }
    
    if test_id == "raise_building_by_id":
        result.update(_validate_raise_building(setup_key, output_text, dataset_path, expected_raise_spec))
    elif test_id == "add_building":
        result.update(_validate_add_building(setup_key, output_text, dataset_path, expected_building_spec))
    else:
        result["reason"] = f"Unknown test_id: {test_id}"
    
    return result


# ============================================================================
# ERWEITERTE VALIDIERUNG: CityJSON Integrität, Geometrie, Import
# ============================================================================

def validate_cityjson_structure(dataset: dict) -> dict:
    """
    Validiert die grundlegende CityJSON-Struktur.

    Prüft:
    - type == "CityJSON"
    - version vorhanden
    - CityObjects ist ein dict
    - vertices ist eine Liste von [x,y,z] Tripeln
    - Jedes CityObject hat "type" und "geometry"

    Returns:
        dict mit "valid" (bool), "errors" (list), "warnings" (list)
    """
    result = {"valid": True, "errors": [], "warnings": []}

    # Grundstruktur
    if not isinstance(dataset, dict):
        result["errors"].append("dataset_not_dict")
        result["valid"] = False
        return result

    if dataset.get("type") != "CityJSON":
        result["errors"].append(f"type_not_cityjson: {dataset.get('type')}")
        result["valid"] = False

    if "version" not in dataset:
        result["errors"].append("version_missing")
        result["valid"] = False

    # CityObjects
    city_objects = dataset.get("CityObjects")
    if not isinstance(city_objects, dict):
        result["errors"].append("CityObjects_not_dict")
        result["valid"] = False
        return result

    for cid, cobj in city_objects.items():
        if not isinstance(cobj, dict):
            result["errors"].append(f"CityObject_{cid}_not_dict")
            result["valid"] = False
            continue
        if "type" not in cobj:
            result["errors"].append(f"CityObject_{cid}_no_type")
            result["valid"] = False
        if "geometry" not in cobj:
            result["warnings"].append(f"CityObject_{cid}_no_geometry")

    # Vertices
    vertices = dataset.get("vertices")
    if not isinstance(vertices, list):
        result["errors"].append("vertices_not_list")
        result["valid"] = False
    else:
        for i, v in enumerate(vertices):
            if not isinstance(v, list) or len(v) != 3:
                result["errors"].append(f"vertex_{i}_invalid: expected [x,y,z]")
                result["valid"] = False
                if len(result["errors"]) > 20:
                    result["errors"].append("...truncated")
                    break

    return result


def validate_building_geometry(building: dict, dataset: dict) -> dict:
    """
    Validiert die Geometrie eines einzelnen Gebäudes.

    Prüft:
    - geometry ist eine Liste mit mind. 1 Element
    - Jede Geometrie hat type, lod, boundaries
    - Alle Vertex-Indizes in boundaries referenzieren gültige Vertices
    - boundaries ist nicht leer

    Returns:
        dict mit "valid" (bool), "errors" (list), "warnings" (list),
        "geometry_count", "vertex_indices_count", "invalid_indices_count"
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "geometry_count": 0,
        "vertex_indices_count": 0,
        "invalid_indices_count": 0,
    }

    geometry_list = building.get("geometry", [])
    if not isinstance(geometry_list, list) or len(geometry_list) == 0:
        result["errors"].append("no_geometry")
        result["valid"] = False
        return result

    result["geometry_count"] = len(geometry_list)
    num_vertices = len(dataset.get("vertices", []))

    for gi, geom in enumerate(geometry_list):
        if not isinstance(geom, dict):
            result["errors"].append(f"geometry_{gi}_not_dict")
            result["valid"] = False
            continue

        geom_type = geom.get("type")
        if geom_type not in ("Solid", "MultiSurface", "CompositeSurface",
                             "MultiSolid", "CompositeSolid"):
            result["warnings"].append(f"geometry_{gi}_unusual_type: {geom_type}")

        if "lod" not in geom:
            result["warnings"].append(f"geometry_{gi}_no_lod")

        boundaries = geom.get("boundaries")
        if boundaries is None or (isinstance(boundaries, list) and len(boundaries) == 0):
            result["errors"].append(f"geometry_{gi}_empty_boundaries")
            result["valid"] = False
            continue

        # Sammle alle Vertex-Indizes rekursiv
        all_indices = []

        def _collect_indices(obj):
            if isinstance(obj, list):
                for item in obj:
                    _collect_indices(item)
            elif isinstance(obj, int):
                all_indices.append(obj)

        _collect_indices(boundaries)
        result["vertex_indices_count"] += len(all_indices)

        # Prüfe ob alle Indizes gültig sind
        invalid = [idx for idx in all_indices if idx < 0 or idx >= num_vertices]
        result["invalid_indices_count"] += len(invalid)

        if invalid:
            result["errors"].append(
                f"geometry_{gi}_invalid_vertex_indices: {len(invalid)} of {len(all_indices)} "
                f"(range 0-{num_vertices - 1}, got {invalid[:5]}{'...' if len(invalid) > 5 else ''})"
            )
            result["valid"] = False

        # Semantics prüfen (optional, aber wertvoll)
        semantics = geom.get("semantics")
        if semantics:
            surfaces = semantics.get("surfaces", [])
            values = semantics.get("values")
            if surfaces:
                result["has_semantics"] = True
                surface_types = [s.get("type") for s in surfaces if isinstance(s, dict)]
                result["semantic_surface_types"] = surface_types
            else:
                result["has_semantics"] = False
        else:
            result["has_semantics"] = False

    return result


def validate_citydpc_import(dataset_path: Path) -> dict:
    """
    Versucht das Dataset mit citydpc zu laden.

    Dies ist der ultimative Integrationstest:
    - Kann die Library das File parsen?
    - Wird das spezifische Gebäude erkannt?
    - Hat es Geometrie-Informationen?

    Returns:
        dict mit "importable" (bool), "building_count", "error", "building_ids"
    """
    result = {
        "importable": False,
        "building_count": 0,
        "error": None,
        "building_ids": [],
    }

    try:
        from citydpc import Dataset
        from citydpc.core import input as citydpc_input
        from citydpc.core.input.cityjsonInput import load_buildings_from_json_file

        d = Dataset()
        citydpc_input.set_roof_volume_calculation(False)
        load_buildings_from_json_file(d, str(dataset_path))

        building_ids = list(d.buildings.keys()) if hasattr(d, 'buildings') else []
        result["importable"] = True
        result["building_count"] = len(building_ids)
        result["building_ids"] = building_ids

    except ImportError:
        result["error"] = "citydpc_not_installed"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-500:]

    return result


def validate_building_displayable(building_id: str, dataset_path: Path) -> dict:
    """
    Prüft ob ein spezifisches Gebäude von citydpc korrekt geladen und
    'angezeigt' werden kann (d.h. Geometrie-Daten korrekt interpretierbar).

    Returns:
        dict mit "displayable" (bool), "has_surfaces", "surface_count",
        "has_roof", "has_wall", "has_ground", "error"
    """
    result = {
        "displayable": False,
        "has_surfaces": False,
        "surface_count": 0,
        "has_roof": False,
        "has_wall": False,
        "has_ground": False,
        "error": None,
    }

    try:
        from citydpc import Dataset
        from citydpc.core import input as citydpc_input
        from citydpc.core.input.cityjsonInput import load_buildings_from_json_file

        d = Dataset()
        citydpc_input.set_roof_volume_calculation(False)
        load_buildings_from_json_file(d, str(dataset_path))

        if building_id not in d.buildings:
            result["error"] = f"building_not_found_in_citydpc: {building_id}"
            return result

        building = d.buildings[building_id]
        result["displayable"] = True

        # Prüfe Oberflächen
        all_surfaces = []
        if hasattr(building, 'get_surfaces'):
            try:
                roof = building.get_surfaces(["RoofSurface"])
                wall = building.get_surfaces(["WallSurface"])
                ground = building.get_surfaces(["GroundSurface"])
                result["has_roof"] = len(roof) > 0
                result["has_wall"] = len(wall) > 0
                result["has_ground"] = len(ground) > 0
                all_surfaces = roof + wall + ground
                result["surface_count"] = len(all_surfaces)
                result["has_surfaces"] = len(all_surfaces) > 0
            except Exception as e:
                result["surface_error"] = f"{type(e).__name__}: {e}"

        # Prüfe Höhe
        if hasattr(building, 'measuredHeight') and building.measuredHeight is not None:
            result["citydpc_measuredHeight"] = building.measuredHeight

    except ImportError:
        result["error"] = "citydpc_not_installed"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-500:]

    return result


# ============================================================================
# STATE-CHANGE VALIDIERUNG
# ============================================================================

def _validate_raise_building(
    setup_key: str,
    output_text: str,
    dataset_path: Path,
    expected_spec: dict | None
) -> dict:
    """Validiert die raise_building_by_id Modifikation."""
    
    if not expected_spec:
        return {"reason": "No expected_raise_spec provided", "accuracy": 0.0}
    
    target_id = expected_spec.get("target_building_id")
    expected_new_height = expected_spec.get("new_height")
    expected_old_height = expected_spec.get("old_height")
    
    checks = {
        "dataset_modified": False,
        "building_found": False,
        "height_correct": False,
        "output_mentions_id": target_id.lower() in output_text.lower() if target_id else False,
        "output_mentions_new_height": str(expected_new_height) in output_text if expected_new_height else False,
    }
    
    # 1. Versuche Dataset zu lesen (für MCP/Code-Interpreter/Patched-No-Tools/CI-Fallback)
    if setup_key in ["mcp", "code_interpreter", "patched_no_tools", "ci_fallback"]:
        try:
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                
                city_objects = dataset.get("CityObjects", {})
                if target_id in city_objects:
                    checks["building_found"] = True
                    building = city_objects[target_id]
                    actual_height = building.get("attributes", {}).get("measuredHeight")
                    checks["actual_height"] = actual_height
                    
                    if actual_height is not None and expected_new_height is not None:
                        # Toleranz: 0.01m
                        if abs(actual_height - expected_new_height) < 0.01:
                            checks["dataset_modified"] = True
                            checks["height_correct"] = True
        except Exception as e:
            checks["dataset_read_error"] = str(e)
    
    # 2. Versuche aus Output zu extrahieren (für alle Setups)
    extracted = extract_city_object_from_text(output_text)
    if extracted:
        city_obj = extracted.get("extracted_object", {})
        attrs = city_obj.get("attributes", {})
        output_height = attrs.get("measuredHeight")
        
        checks["output_contains_city_object"] = True
        checks["output_height"] = output_height
        
        if output_height is not None and expected_new_height is not None:
            if abs(output_height - expected_new_height) < 0.01:
                checks["output_height_correct"] = True
    else:
        checks["output_contains_city_object"] = False
    
    # 3. Dataset-Integrität prüfen (CityJSON Struktur + citydpc Import)
    integrity_results = {}
    if setup_key in ["mcp", "code_interpreter", "patched_no_tools", "ci_fallback"]:
        try:
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    full_dataset = json.load(f)
                # CityJSON Strukturvalidierung
                structure_check = validate_cityjson_structure(full_dataset)
                checks["cityjson_valid"] = structure_check["valid"]
                integrity_results["cityjson_structure"] = structure_check

                # citydpc Import-Test
                import_check = validate_citydpc_import(dataset_path)
                checks["citydpc_importable"] = import_check["importable"]
                integrity_results["citydpc_import"] = import_check
        except Exception as e:
            checks["integrity_error"] = str(e)
    
    # 4. Accuracy berechnen
    # Für MCP/Code-Interpreter/Patched-No-Tools/CI-Fallback: Dataset-Prüfung hat Priorität
    # Für No-Tools: Nur Output-Prüfung möglich
    
    if setup_key == "no_tools":
        # Strikt dataset-basiert: Ohne angewendeten Patch gilt no_tools nicht als
        # tatsächliche Modifikation. (patched_no_tools wird separat dateibasiert geprüft)
        score = 0.0
        modification_verified = False
        checks["dataset_only_mode"] = True
        checks["dataset_change_required"] = True
    else:
        # MCP/Code-Interpreter/Patched-No-Tools/CI-Fallback: Dataset muss modifiziert sein
        score = 0.0
        if checks.get("building_found"):
            score += 0.25
        if checks.get("dataset_modified"):
            score += 0.5
        if checks.get("height_correct"):
            score += 0.25
        
        modification_verified = checks.get("height_correct", False)
    
    return {
        "validated": True,
        "modification_verified": modification_verified,
        "accuracy": min(score, 1.0),
        "checks": checks,
        "integrity": integrity_results,
        "expected": {
            "building_id": target_id,
            "old_height": expected_old_height,
            "new_height": expected_new_height,
        }
    }


def _validate_add_building(
    setup_key: str,
    output_text: str,
    dataset_path: Path,
    expected_spec: dict | None
) -> dict:
    """
    Validiert die add_building Erstellung.
    
    Prüft nicht nur Attribute, sondern auch:
    - CityJSON Struktur-Validität des gesamten Datasets
    - Geometrie-Validierung (Vertex-Referenzen, Boundaries)
    - citydpc Import-Test (Dataset ladbar durch Library)
    - Gebäude-Darstellbarkeit (Oberflächen vorhanden)
    """
    
    if not expected_spec:
        return {"reason": "No expected_building_spec provided", "accuracy": 0.0}
    
    expected_id = expected_spec.get("id")
    expected_height = expected_spec.get("measuredHeight")
    expected_storeys = expected_spec.get("storeysAboveGround")
    expected_roof_type = expected_spec.get("roofType")
    expected_function = expected_spec.get("function")
    expected_coords = expected_spec.get("groundCoordinates", [])
    
    checks = {
        "dataset_modified": False,
        "building_found": False,
        "id_correct": False,
        "height_correct": False,
        "storeys_correct": False,
        "roof_type_correct": False,
        "function_correct": False,
        "has_geometry": False,
        # Erweiterte Checks
        "geometry_valid": False,        # Vertex-Indizes gültig, Boundaries vorhanden
        "cityjson_valid": False,        # CityJSON Struktur valide
        "citydpc_importable": False,    # citydpc kann Dataset laden
        "building_displayable": False,  # Gebäude hat Oberflächen in citydpc
        "output_mentions_id": expected_id.lower() in output_text.lower() if expected_id else False,
    }
    
    integrity_results = {}
    building_from_dataset = None
    building_from_output = None
    full_dataset = None
    
    # 1. Versuche Dataset zu lesen (für MCP/Code-Interpreter/Patched-No-Tools/CI-Fallback)
    if setup_key in ["mcp", "code_interpreter", "patched_no_tools", "ci_fallback"]:
        try:
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    full_dataset = json.load(f)
                
                city_objects = full_dataset.get("CityObjects", {})
                
                # Suche nach dem erwarteten Gebäude
                if expected_id in city_objects:
                    checks["building_found"] = True
                    checks["id_correct"] = True
                    checks["dataset_modified"] = True
                    building_from_dataset = city_objects[expected_id]
                else:
                    # Vielleicht hat das LLM eine andere ID verwendet?
                    for bid, bdata in city_objects.items():
                        if bid.startswith("EVAL_") or "TEST" in bid.upper():
                            checks["building_found"] = True
                            checks["dataset_modified"] = True
                            building_from_dataset = bdata
                            checks["actual_id"] = bid
                            break
        except Exception as e:
            checks["dataset_read_error"] = str(e)
    
    # 2. Versuche aus Output zu extrahieren (für alle Setups)
    extracted = extract_city_object_from_text(output_text)
    if extracted:
        checks["output_contains_city_object"] = True
        building_from_output = extracted.get("extracted_object", {})
        if extracted.get("has_id_wrapper"):
            checks["output_building_id"] = extracted.get("building_id")
    else:
        checks["output_contains_city_object"] = False
    
    # 3. Validiere Gebäude-Attribute (aus Dataset oder Output)
    building_to_check = building_from_dataset or building_from_output
    
    if building_to_check:
        attrs = building_to_check.get("attributes", {})
        
        # Höhe prüfen
        actual_height = attrs.get("measuredHeight")
        if actual_height is not None and expected_height is not None:
            checks["height_correct"] = abs(actual_height - expected_height) < 0.1
            checks["actual_height"] = actual_height
        
        # Stockwerke prüfen
        actual_storeys = attrs.get("storeysAboveGround")
        if actual_storeys is not None and expected_storeys is not None:
            checks["storeys_correct"] = actual_storeys == expected_storeys
            checks["actual_storeys"] = actual_storeys
        
        # Dachtyp prüfen
        actual_roof = attrs.get("roofType")
        if actual_roof and expected_roof_type:
            checks["roof_type_correct"] = str(actual_roof) == str(expected_roof_type)
            checks["actual_roof_type"] = actual_roof
        
        # Funktion prüfen
        actual_function = attrs.get("function")
        if actual_function and expected_function:
            checks["function_correct"] = str(actual_function) == str(expected_function)
            checks["actual_function"] = actual_function
        
        # Geometrie prüfen (Basis)
        geometry = building_to_check.get("geometry", [])
        checks["has_geometry"] = len(geometry) > 0
        if geometry:
            checks["geometry_type"] = geometry[0].get("type") if geometry else None
            checks["geometry_lod"] = geometry[0].get("lod") if geometry else None
    
    # ================================================================
    # 4. ERWEITERTE VALIDIERUNG: Integrität, Geometrie, Import
    # ================================================================
    
    # 4a. CityJSON Struktur-Validierung
    if full_dataset is not None:
        structure_check = validate_cityjson_structure(full_dataset)
        checks["cityjson_valid"] = structure_check["valid"]
        integrity_results["cityjson_structure"] = structure_check
    
    # 4b. Geometrie-Validierung des neuen Gebäudes
    if building_from_dataset is not None and full_dataset is not None:
        geom_check = validate_building_geometry(building_from_dataset, full_dataset)
        checks["geometry_valid"] = geom_check["valid"]
        integrity_results["geometry_validation"] = geom_check
    elif building_to_check and full_dataset is not None:
        # Fallback: prüfe building_from_output mit dem Dataset
        geom_check = validate_building_geometry(building_to_check, full_dataset)
        checks["geometry_valid"] = geom_check["valid"]
        integrity_results["geometry_validation"] = geom_check
    
    # 4c. citydpc Import-Test (kann das gesamte Dataset geladen werden?)
    if setup_key in ["mcp", "code_interpreter", "patched_no_tools", "ci_fallback"]:
        if dataset_path.exists():
            import_check = validate_citydpc_import(dataset_path)
            checks["citydpc_importable"] = import_check["importable"]
            integrity_results["citydpc_import"] = import_check
            
            # 4d. Gebäude-Darstellbarkeit (hat das Gebäude Oberflächen in citydpc?)
            building_id_to_check = expected_id
            if not checks.get("id_correct") and checks.get("actual_id"):
                building_id_to_check = checks["actual_id"]
            
            if import_check["importable"] and building_id_to_check:
                display_check = validate_building_displayable(building_id_to_check, dataset_path)
                checks["building_displayable"] = display_check["displayable"]
                if display_check.get("has_surfaces"):
                    checks["building_has_surfaces"] = True
                    checks["building_surface_count"] = display_check.get("surface_count", 0)
                    checks["building_has_roof"] = display_check.get("has_roof", False)
                    checks["building_has_wall"] = display_check.get("has_wall", False)
                    checks["building_has_ground"] = display_check.get("has_ground", False)
                integrity_results["building_displayable"] = display_check
    
    # 5. Accuracy berechnen
    if setup_key == "no_tools":
        # Strikt dataset-basiert: Ohne angewendeten Patch gilt no_tools nicht als
        # tatsächliche Modifikation. (patched_no_tools wird separat dateibasiert geprüft)
        score = 0.0
        modification_verified = False
        checks["dataset_only_mode"] = True
        checks["dataset_change_required"] = True
    else:
        # MCP/Code-Interpreter/Patched-No-Tools/CI-Fallback: Dataset muss modifiziert sein
        score = 0.0
        weights = {
            "building_found": 0.1,
            "dataset_modified": 0.1,
            "id_correct": 0.1,
            "height_correct": 0.1,
            "storeys_correct": 0.05,
            "roof_type_correct": 0.05,
            "function_correct": 0.05,
            "has_geometry": 0.1,
            "geometry_valid": 0.1,        # NEU: Vertex-Indizes korrekt
            "cityjson_valid": 0.1,        # NEU: Dataset valide
            "citydpc_importable": 0.1,    # NEU: Library kann importieren
            "building_displayable": 0.05, # NEU: Gebäude darstellbar
        }
        for check, weight in weights.items():
            if checks.get(check):
                score += weight
        
        modification_verified = checks.get("dataset_modified", False) and checks.get("has_geometry", False)
    
    return {
        "validated": True,
        "modification_verified": modification_verified,
        "accuracy": min(score, 1.0),
        "checks": checks,
        "integrity": integrity_results,
        "expected": {
            "building_id": expected_id,
            "measuredHeight": expected_height,
            "storeysAboveGround": expected_storeys,
            "roofType": expected_roof_type,
            "function": expected_function,
        },
        "building_source": "dataset" if building_from_dataset else ("output" if building_from_output else "none")
    }
