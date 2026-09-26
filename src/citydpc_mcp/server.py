"""CityDPC MCP server: CityJSON/CityGML building tools over stdio.

Tool names, signatures and (German) descriptions are exactly those used in the
paper evaluation; do not change them without re-running the benchmark.
"""
from pathlib import Path
from typing import Optional
import argparse
import os
import math
import copy
from datetime import datetime
from citydpc.util.envelope import update_min_max_from_surface

from citydpc.dataset import Dataset
from citydpc.core.input.citygmlInput import load_buildings_from_xml_file
from citydpc.core.input.cityjsonInput import load_buildings_from_json_file
from citydpc.core import input
from citydpc.core.object.abstractBuilding import AbstractBuilding
from citydpc.core.object.building import Building
from citydpc.core.object.address import AddressCollection
from citydpc.core.output.citygmlOutput import write_citygml_file
from citydpc.core.output.cityjsonOutput import write_cityjson_file
from citydpc.tools import cityBIT, cityATB, partywall
input.set_roof_volume_calculation(True)

from fastmcp import FastMCP
from mcp.types import Icon


dataset: Optional[Dataset] = None
dataset_path: Optional[Path] = None
is_filtered: bool = False
snapshots: list[dict] = []


# Directory with the *.json / *.gml datasets the tools may load and overwrite.
# Defaults to the working directory of the server process.
DATASET_DIR = Path(os.getenv("MCP_DATASET_DIR", ".")).expanduser().resolve()


mcp = FastMCP(
    name="cityDPC",
    instructions="You are a helpful assistant that can help with using tools provided by CityDPC."
)


# ===============================================================================
# Hilfs- und Systemfunktionen
# ===============================================================================

def _get_available_datasets() -> list[str]:
    """Gibt eine Liste aller verfügbaren GML- und JSON-Dateien im Dataset-Verzeichnis zurück.
    
    Returns:
        list[str]: Liste der Dateinamen
    """
    gml_files = sorted(DATASET_DIR.glob("*.gml"))
    json_files = sorted(DATASET_DIR.glob("*.json"))
    files = gml_files + json_files
    
    if not files:
        return []
    
    return [f.name for f in files]


def _isgml_file(filename: str) -> bool:
    """Prüft ob eine Datei eine GML-Datei ist.
    
    Args:
        filename: Dateiname
    
    Returns:
        bool: True wenn GML-Datei
    """
    return filename.endswith(".gml")


def _isjson_file(filename: str) -> bool:
    """Prüft ob eine Datei eine JSON-Datei ist.
    
    Args:
        filename: Dateiname
    
    Returns:
        bool: True wenn JSON-Datei
    """
    return filename.endswith(".json")


def _get_target_path(target_format: Optional[str]) -> Path:
    if dataset_path is None:
        raise ValueError("Kein Dataset geladen.")

    if target_format is None:
        return dataset_path
    if target_format == "gml":
        return dataset_path.with_suffix(".gml")
    if target_format == "json":
        return dataset_path.with_suffix(".json")

    raise ValueError(
        f"Ungueltiges Zielformat: {target_format}. Erlaubt: 'gml', 'json'."
    )


def _write_dataset(dataset: Dataset, target_format: Optional[str] = None) -> Path:
    """Schreibt ein Dataset in eine GML- oder JSON-Datei.
    
    Args:
        dataset: Das zu speichernde Dataset
        target_format: Optionales Zielformat ("gml" oder "json"). Wenn None, wird das ursprüngliche Format beibehalten.
    """
    target_path = _get_target_path(target_format)

    if target_format == "gml":
        write_citygml_file(dataset, str(target_path), version="2.0")
    elif target_format == "json":
        write_cityjson_file(dataset, str(target_path), version="2.0")
    else:
        if _isgml_file(target_path.name):
            write_citygml_file(dataset, str(target_path), version="2.0")
        elif _isjson_file(target_path.name):
            write_cityjson_file(dataset, str(target_path), version="2.0")
        else:
            raise ValueError(f"Ungültige Dateiendung: {target_path}")

    return target_path


def _json_safe(value):
    """Convert CityDPC/lxml helper values into MCP-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _serialize_building(building: Building) -> dict:
    """Serialisiert ein Gebäude-Objekt in ein Dictionary.

    Args:
        building: Das zu serialisierende Gebäude

    Returns:
        dict: Dictionary mit allen Gebäude-Attributen
    """
    min_coordinates, max_coordinates = _get_building_bounding_box(building)
    return {
        "gml_id": _json_safe(building.gml_id),
        "measured_height": _json_safe(building.measuredHeight),
        "roof_type": _json_safe(building.roofType),
        "roof_height": _json_safe(building.roof_height),
        "roof_volume": _json_safe(building.roof_volume),
        "lod": _json_safe(building.lod),
        "function": _json_safe(building.function),
        "usage": _json_safe(building.usage),
        "year_of_construction": _json_safe(building.yearOfConstruction),
        "storeys_above_ground": _json_safe(building.storeysAboveGround),
        "storeys_below_ground": _json_safe(building.storeysBelowGround),
        "storey_heights_above_ground": _json_safe(building.storeyHeightsAboveGround),
        "storey_heights_below_ground": _json_safe(building.storeyHeightsBelowGround),
        "creation_date": _json_safe(building.creationDate),
        "min_coordinates": min_coordinates,
        "max_coordinates": max_coordinates,
        "address": _serialize_addresses(building.addressCollection),
    }


def _serialize_dataset(dataset: Dataset) -> dict:
    """Serialisiert ein Dataset-Objekt in ein Dictionary.

    Args:
        dataset: Das zu serialisierende Dataset

    Returns:
        dict: Dictionary mit Dataset-Informationen
    """
    return {
        "building_count": len(dataset),
        "minimum": dataset._minimum,
        "maximum": dataset._maximum,
        "filepath": str(dataset_path),
        "title": dataset.title,
    }


def _serialize_addresses(address_collection: Optional[AddressCollection],) -> list[dict]:
    """Serialisiert eine AddressCollection zu einer Liste von Dictionaries.

    Args:
        address_collection: AddressCollection mit Address-Einträgen

    Returns:
        list[dict]: Liste der serialisierten Adressen
    """
    if not address_collection:
        return []

    raw_addresses = address_collection.get_adresses() or []
    addresses = []
    for address in raw_addresses:
        if address is None:
            continue
        address_dict = {
            "gml_id": _json_safe(address.gml_id),
            "countryName": _json_safe(address.countryName),
            "locality_type": _json_safe(address.locality_type),
            "localityName": _json_safe(address.localityName),
            "thoroughfare_type": _json_safe(address.thoroughfare_type),
            "thoroughfareNumber": _json_safe(address.thoroughfareNumber),
            "thoroughfareName": _json_safe(address.thoroughfareName),
            "postalCodeNumber": _json_safe(address.postalCodeNumber),
        }
        addresses.append(address_dict)
    return addresses


def _ensure_dataset_loaded() -> Dataset:
    """Stellt sicher, dass ein Dataset geladen ist. Lädt das Standard-Dataset falls nötig.

    Returns:
        Dataset: Das geladene Dataset
    """
    global dataset
    if dataset is None:
        dataset = Dataset()
        load_buildings_from_xml_file(dataset, str(dataset_path), use_multiprocessing=False)
    return dataset


def _get_building_bounding_box(building: Building) -> tuple:
    """Berechnet die Bounding Box eines Gebäudes aus allen Oberflächen.
    
    Args:
        building: Das Gebäude
    
    Returns:
        tuple: (min_coordinates, max_coordinates)
    """
    min_coordinates = [math.inf, math.inf, math.inf]
    max_coordinates = [-math.inf, -math.inf, -math.inf]
    surfaces = building.get_surfaces()
    
    for surface in surfaces:
        min_coordinates, max_coordinates = update_min_max_from_surface(
            min_coordinates, max_coordinates, surface
        )
    
    return min_coordinates, max_coordinates


def _create_snapshot(description: str) -> dict:
    """Interne Hilfsfunktion zum Erstellen eines Snapshots.

    Args:
        description: Beschreibung des Snapshots

    Returns:
        dict: Informationen über den erstellten Snapshot
    """
    dataset = _ensure_dataset_loaded()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_index = len(snapshots)

    snapshot = {
        'index': snapshot_index,
        'timestamp': timestamp,
        'description': description or f"Snapshot {snapshot_index}",
        'dataset_snapshot': copy.deepcopy(dataset),
        'building_count': len(dataset),
    }

    snapshots.append(snapshot)

    return {
        "index": snapshot_index,
        "timestamp": timestamp,
        "description": snapshot['description'],
        "building_count": len(dataset),
        "message": f"Snapshot {snapshot_index} erstellt: {snapshot['description']}"
    }


# ===============================================================================
# Dataset-Management
# ===============================================================================

@mcp.tool()
def list_datasets() -> list[str]:
    """Listet alle verfügbaren GML- oder JSON-Datasets im Data-Verzeichnis auf.
    
    Returns:
        list[str]: Liste aller verfügbaren Dataset-Dateinamen
    """
    return _get_available_datasets()


@mcp.tool()
def load_dataset(filename: str) -> str:
    """Lädt ein Dataset aus einer GML- oder JSON-Datei.

    Erstellt automatisch einen Snapshot des ursprünglichen Zustands.

    Args:
        filename: Der Name der GML- oder JSON-Datei (z.B. 'EssenExample.gml')

    Returns:
        str: Erfolgsmeldung oder Fehlermeldung
    """
    global dataset_path, dataset, is_filtered, snapshots

    target_path = DATASET_DIR / filename

    if not target_path.exists():
        available = list_datasets()
        return f"Fehler: Datei '{filename}' nicht gefunden.\nVerfügbar: {', '.join(available)}"

    dataset_path = target_path
    is_filtered = False
    snapshots.clear()

    try:
        dataset = Dataset()
        if _isgml_file(filename):
            load_buildings_from_xml_file(dataset, str(dataset_path), use_multiprocessing=False)
        elif _isjson_file(filename):
            load_buildings_from_json_file(dataset, str(dataset_path))
        else:
            raise ValueError(f"Ungültige Dateiendung: {dataset_path}")

        _create_snapshot("Initial Load")

        return f"Erfolg: '{filename}' geladen ({len(dataset)} Gebäude)."
    except Exception as e:
        return f"Fehler beim Laden von '{filename}': {str(e)}"


@mcp.tool()
def filter_dataset(addressRestriciton: dict = None, borderCoordinates: list = None) -> dict:
    """Filtert das Dataset nach Adressen oder Koordinaten und setzt das gefilterte Dataset als aktives Dataset.

    Das gefilterte Dataset wird zum neuen aktiven Dataset. Alle nachfolgenden Tool-Aufrufe arbeiten
    dann nur mit den gefilterten Gebäuden. Mit load_dataset() kann jederzeit das Original neu geladen werden.

    WICHTIG: Nach dem Filtern können keine Änderungen mehr gespeichert werden, um das Original zu schützen.
    Die History wird geleert!

    Args:
        addressRestriciton: Dictionary mit Adressbeschränkungen (z.B. {"thoroughfareName": "Stakenholt"})
        borderCoordinates: Liste von Koordinaten für einen rechteckigen Bereich (z.B. [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])

    Returns:
        dict: Informationen zum gefilterten Dataset (building_count, minimum, maximum, filepath, title)
    """
    global dataset, is_filtered, snapshots
    dataset = _ensure_dataset_loaded()
    dataset = cityATB.search_dataset(dataset, borderCoordinates, addressRestriciton, inplace=False)
    is_filtered = True
    snapshots.clear()
    return _serialize_dataset(dataset)


# ===============================================================================
# Analyse- und Abfragefunktionen
# ===============================================================================

@mcp.tool()
def analyse_dataset() -> dict:
    """Analysiert das Dataset und gibt detaillierte Informationen über die Gebäude zurück.
    
    Returns:
        dict: Analyse-Ergebnisse mit Informationen über GML-Version, CRS, LoD, Anzahl Gebäude etc.
    """
    dataset = _ensure_dataset_loaded()
    return cityATB.analysis(dataset)


@mcp.tool()
def number_of_buildings() -> int:
    """Gibt die Anzahl der Gebäude im aktuellen Dataset zurück.
    
    Returns:
        int: Anzahl der Gebäude
    """
    dataset = _ensure_dataset_loaded()
    return len(dataset)


@mcp.tool()
def get_all_buildings() -> list[dict]:
    """Gibt Informationen zu allen Gebäuden im Dataset zurück.
    
    Returns:
        list[dict]: Liste mit Dictionaries aller Gebäude und deren Attributen
    """
    dataset = _ensure_dataset_loaded()
    return [_serialize_building(building) for building in dataset.buildings.values()]


@mcp.tool()
def get_buiding_Id_list() -> list[str]:
    """Gibt eine Liste aller Gebäude-IDs im Dataset zurück.
    
    Returns:
        list[str]: Liste aller gml:id Werte der Gebäude
    """
    dataset = _ensure_dataset_loaded()
    return list(dataset.buildings.keys())


@mcp.tool()
def get_party_walls() -> list:
    """Findet alle angrenzenden Wände (Party Walls) zwischen Gebäuden im Dataset.
    
    Returns:
        list: Liste aller erkannten Party Walls als [id of b0, id of w0, id of b1, id of w1, area, collision coordinates]
    """
    dataset = _ensure_dataset_loaded()
    return partywall.get_party_walls(dataset)


# ===============================================================================
# Gebäudespezifische Abfragen
# ===============================================================================

@mcp.tool()
def get_building_by_id(building_id: str) -> dict:
    """Gibt alle Informationen zu einem spezifischen Gebäude zurück.
    
    Args:
        building_id: Die gml:id des Gebäudes
    
    Returns:
        dict: Dictionary mit allen Gebäude-Attributen
    """
    dataset = _ensure_dataset_loaded()
    
    if building_id not in dataset.buildings:
        raise ValueError(f"Gebäude '{building_id}' nicht im Dataset gefunden.")
    
    building = dataset.buildings[building_id]
    return _serialize_building(building)


@mcp.tool()
def calculate_roof_volume_by_id(building_id: str) -> float:
    """Berechnet das Dachvolumen für ein spezifisches Gebäude.
    
    Args:
        building_id: Die gml:id des Gebäudes
    
    Returns:
        float: Das berechnete Dachvolumen in Kubikmetern
    """
    dataset = _ensure_dataset_loaded()
    building = dataset.buildings.get(building_id)
    building._calc_roof_volume()
    return building.roof_volume or 0.0


# ===============================================================================
# Erstellung und Modifikation von Gebäuden
# ===============================================================================

@mcp.tool()
def create_building(
    id: str,
    groundsCoordinates: list[list[float]],
    groundSurfaceHeight: float,
    geometryHeight: float = None,
    roofType: str = None,
    roofHeight: float = None,
    roofOrientation: int = None,
    lod: int = 2,
    isRoofEdge: bool = False,
) -> dict:
    """Erstellt ein neues Gebäude mit wählbarem Detaillierungsgrad (LoD 0/1/2).
    
    LoD: 0=2D-Grundfläche | 1=3D-Quader | 2=3D mit Dachgeometrie (Standard)
    
    Das Gebäude wird direkt zum aktiven Dataset hinzugefügt (Single Source of Truth).
    Verwende take_snapshot() vor dem Erstellen und save_dataset() zum Speichern.
    
    Args:
        id: Eindeutige Gebäude-ID
        groundsCoordinates: 2D-Grundriss als Liste von Koordinaten [[x1,y1], [x2,y2], ..., [x1,y1]]
        groundSurfaceHeight: Bodenhöhe in Metern
        geometryHeight: Gebäudehöhe in Metern (erforderlich für LoD 1+2)
        roofType: Dachtyp (für LoD 2): "1000"=Flach, "1010"=Pult, "1020"=Pult versetzt, "1030"=Sattel, "1040"=Walm, "1070"=Zelt
        roofHeight: Dachhöhe in Metern (für LoD 2, außer "1000")
        roofOrientation: Dachausrichtung als Koordinaten-Index (für "1010","1020","1030")
        lod: Detaillierungsgrad 0-2 (Standard: 2)
        isRoofEdge: Für LoD 0, ob Koordinaten Dachkante sind (Standard: False)
    
    Returns:
        dict: Das erstellte Gebäude mit allen Attributen und Bounding Box
    """
    if is_filtered:
        raise ValueError("Erstellen nicht möglich: Das Dataset ist gefiltert. Bitte erst load_dataset() aufrufen, um das vollständige Dataset zu laden.")

    dataset = _ensure_dataset_loaded()

    if id in dataset.buildings:
        raise ValueError(f"Gebäude mit ID '{id}' existiert bereits im Dataset.")
    
    if lod == 0:
        building = cityBIT.create_LoD0_building(
            id, groundsCoordinates, groundSurfaceHeight, isRoofEdge
        )
    elif lod == 1:
        if geometryHeight is None:
            raise ValueError("geometryHeight ist erforderlich für LoD1-Gebäude")
        building = cityBIT.create_LoD1_building(
            id, groundsCoordinates, groundSurfaceHeight, geometryHeight
        )
    elif lod == 2:
        if geometryHeight is None:
            raise ValueError("geometryHeight ist erforderlich für LoD2-Gebäude")
        if roofType is None:
            raise ValueError("roofType ist erforderlich für LoD2-Gebäude")
        building = cityBIT.create_LoD2_building(
            id, groundsCoordinates, groundSurfaceHeight, geometryHeight,
            roofType, roofHeight, roofOrientation
        )
    else:
        raise ValueError(f"Ungültiger LoD-Wert: {lod}. Erlaubt sind: 0, 1, 2")
    
    dataset.add_building(building)
    return _serialize_building(building)


@mcp.tool()
def remove_building_from_dataset(building_id: str) -> dict:
    """Entfernt ein Gebäude aus dem Dataset.
    
    Die Änderung wird direkt im Dataset vorgenommen (Single Source of Truth).
    Verwende take_snapshot() vor wichtigen Änderungen und save_dataset() zum Speichern.
    
    Args:
        building_id: Die gml:id des zu entfernenden Gebäudes
    
    Returns:
        dict: Bestätigungsmeldung mit Anzahl verbleibender Gebäude
    """
    if is_filtered:
        raise ValueError("Entfernen nicht möglich: Das Dataset ist gefiltert. Bitte erst load_dataset() aufrufen, um das vollständige Dataset zu laden.")
    
    dataset = _ensure_dataset_loaded()
    
    if building_id not in dataset.buildings:
        raise ValueError(f"Gebäude '{building_id}' nicht im Dataset gefunden.")
    
    keep_ids = [bid for bid in dataset.buildings.keys() if bid != building_id]
    dataset.reduce(keep_ids)
    
    return {
        "message": f"Gebäude '{building_id}' wurde aus dem Dataset entfernt. Verbleibende Gebäude: {len(dataset)}"
    }


@mcp.tool()
def enrich_building(
    building_id: str,
    measured_height: float = None,
    roof_type: str = None,
    roof_height: float = None,
    function: str = None,
    usage: str = None,
    year_of_construction: int = None,
    storeys_above_ground: int = None,
    storeys_below_ground: int = None,
    creation_date: str = None,
) -> dict:
    """Reichert ein Gebäude mit semantischen Informationen an.

    Die Änderungen werden direkt im Dataset vorgenommen (Single Source of Truth).
    Verwende take_snapshot() vor wichtigen Änderungen und save_dataset() zum Speichern.

    Args:
        building_id: Die gml:id des Gebäudes
        measured_height: Gemessene Gebäudehöhe in Metern
        roof_type: Dachtyp-Code (z.B. "1000"=Flach, "1030"=Sattel, "1040"=Walm, "3100"=andere)
        roof_height: Dachhöhe in Metern
        function: Gebäudefunktion-Code nach CityGML (z.B. "31001_1010"=Wohngebäude, "31001_2000"=Gewerbe)
        usage: Nutzung des Gebäudes (z.B. "residential", "commercial", "industrial")
        year_of_construction: Baujahr (z.B. 2020)
        storeys_above_ground: Anzahl der Stockwerke über dem Boden
        storeys_below_ground: Anzahl der Stockwerke unter dem Boden (Keller)
        creation_date: Erstellungsdatum des Gebäudes (ISO-Format: "2020-01-15")
    Returns:
        dict: Das angereicherte Gebäude mit allen Attributen
    """
    dataset = _ensure_dataset_loaded()

    if building_id not in dataset.buildings:
        raise ValueError(f"Gebäude '{building_id}' nicht im Dataset gefunden.")

    building = dataset.buildings[building_id]

    attributes = {
        'measuredHeight': measured_height,
        'roofType': roof_type,
        'roof_height': roof_height,
        'function': function,
        'usage': usage,
        'yearOfConstruction': year_of_construction,
        'storeysAboveGround': storeys_above_ground,
        'storeysBelowGround': storeys_below_ground,
        'creationDate': creation_date,
    }

    for attr_name, value in attributes.items():
        if value is not None:
            setattr(building, attr_name, value)

    return _serialize_building(building)


@mcp.tool()
def remove_building_attributes(building_id: str, attributes: list[str]) -> dict:
    """Entfernt Attribute eines Gebäudes (setzt sie auf None).

    Args:
        building_id: Die gml:id des Gebäudes
        attributes: Liste von Attributnamen (gleiche Namen wie in enrich_building,
            z.B. "measuredHeight", "roofType"). Unbekannte Namen werden ignoriert.

    Returns:
        dict: Das aktualisierte Gebäude mit allen Attributen
    """
    dataset = _ensure_dataset_loaded()

    if building_id not in dataset.buildings:
        raise ValueError(f"Gebäude '{building_id}' nicht im Dataset gefunden.")

    if not attributes:
        raise ValueError("Keine Attribute angegeben.")

    building = dataset.buildings[building_id]

    allowed_attributes = {
        "measuredHeight",
        "roofType",
        "roof_height",
        "function",
        "usage",
        "yearOfConstruction",
        "storeysAboveGround",
        "storeysBelowGround",
        "creationDate",
    }

    for attr in attributes:
        if attr in allowed_attributes:
            setattr(building, attr, None)

    return _serialize_building(building)


# ===============================================================================
# Snapshot und Versionsverwaltung
# ===============================================================================

@mcp.tool()
def take_snapshot(description: str = "") -> dict:
    """Erstellt einen Snapshot des aktuellen Dataset-Zustands.
    
    Wie ein Git-Commit: Speichert den aktuellen Zustand für spätere Rollbacks.
    
    Args:
        description: Beschreibung des Snapshots
    
    Returns:
        dict: Informationen über den erstellten Snapshot
    """
    return _create_snapshot(description or "")


@mcp.tool()
def rollback_to_snapshot(snapshot_index: int) -> dict:
    """Stellt einen früheren Dataset-Zustand wieder her.

    Wie Git-Reset: Kehrt zu einem gespeicherten Zustand zurück.
    Alle Änderungen nach diesem Snapshot gehen verloren!

    Args:
        snapshot_index: Index des Snapshots (von get_dataset_history() erhalten)

    Returns:
        dict: Informationen über den Rollback
    """
    global dataset, snapshots

    if snapshot_index < 0 or snapshot_index >= len(snapshots):
        raise ValueError(f"Ungültiger Snapshot-Index: {snapshot_index}. Verfügbare Indizes: 0-{len(snapshots)-1}")

    snapshot = snapshots[snapshot_index]
    dataset = copy.deepcopy(snapshot['dataset_snapshot'])

    snapshots = snapshots[:snapshot_index + 1]

    return {
        "index": snapshot_index,
        "timestamp": snapshot['timestamp'],
        "description": snapshot['description'],
        "building_count": len(dataset),
        "message": f"Dataset wurde zu Snapshot {snapshot_index} zurückgesetzt: {snapshot['description']}"
    }


@mcp.tool()
def save_dataset(description: str = "", target_format: Optional[str] = None) -> dict:
    """Speichert das Dataset dauerhaft in die GML/JSON-Datei.

    Übernimmt alle aktuellen Änderungen permanent in die Datei.

    Args:
        description: Beschreibung des Speicherns
        target_format: Optionales Zielformat ("gml" oder "json"). Wenn None, wird das ursprüngliche Format beibehalten.
    Returns:
        dict: Informationen über den Speichervorgang
    """
    if is_filtered:
        raise ValueError("Speichern nicht möglich: Das Dataset ist gefiltert. Bitte erst load_dataset() aufrufen, um das vollständige Dataset zu laden.")

    global dataset_path
    dataset = _ensure_dataset_loaded()

    target_path = _write_dataset(dataset, target_format)
    if target_path != dataset_path:
        dataset_path = target_path

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "timestamp": timestamp,
        "description": description or "Dataset gespeichert",
        "filepath": str(target_path),
        "building_count": len(dataset),
        "message": f"Dataset erfolgreich in {target_path.name} gespeichert ({len(dataset)} Gebäude)"
    }


@mcp.tool()
def get_dataset_history() -> dict:
    """Zeigt die Historie aller Snapshots und Speicherungen.

    Returns:
        dict: Übersicht über Snapshots, Speicherungen und aktuellen Zustand
    """
    dataset = _ensure_dataset_loaded()

    snapshots_info = []
    for snapshot in snapshots:
        snapshots_info.append({
            "index": snapshot['index'],
            "timestamp": snapshot['timestamp'],
            "description": snapshot['description'],
            "building_count": snapshot['building_count']
        })

    return {
        "current_building_count": len(dataset),
        "current_filepath": str(dataset_path) if dataset_path else None,
        "is_filtered": is_filtered,
        "snapshots": snapshots_info,
        "total_snapshots": len(snapshots)
    }


def main() -> None:
    global DATASET_DIR
    parser = argparse.ArgumentParser(description="CityDPC MCP server (stdio)")
    parser.add_argument(
        "--dataset-dir",
        help="Directory with CityJSON/CityGML files (default: $MCP_DATASET_DIR or the current directory)",
    )
    args = parser.parse_args()
    if args.dataset_dir:
        DATASET_DIR = Path(args.dataset_dir).expanduser().resolve()
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
