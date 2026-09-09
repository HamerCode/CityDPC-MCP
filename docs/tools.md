# MCP tool reference

All 18 tool signatures and original German descriptions from `mcp-server/cityDPC.py`. Parameter spelling is preserved for compatibility. See the [server guide](mcp-server.md) for English usage and state/persistence semantics. The source code is authoritative; tool descriptions are not additional validation guarantees.

## `list_datasets`

```python
list_datasets() -> list[str]
```

Listet alle verfügbaren GML- oder JSON-Datasets im Data-Verzeichnis auf.

Returns:
    list[str]: Liste aller verfügbaren Dataset-Dateinamen

## `load_dataset`

```python
load_dataset(filename: str) -> str
```

Lädt ein Dataset aus einer GML- oder JSON-Datei.

Erstellt automatisch einen Snapshot des ursprünglichen Zustands.

Args:
    filename: Der Name der GML- oder JSON-Datei (z.B. 'EssenExample.gml')

Returns:
    str: Erfolgsmeldung oder Fehlermeldung

## `filter_dataset`

```python
filter_dataset(addressRestriciton: dict=None, borderCoordinates: list=None) -> dict
```

Filtert das Dataset nach Adressen oder Koordinaten und setzt das gefilterte Dataset als aktives Dataset.

Das gefilterte Dataset wird zum neuen aktiven Dataset. Alle nachfolgenden Tool-Aufrufe arbeiten
dann nur mit den gefilterten Gebäuden. Mit load_dataset() kann jederzeit das Original neu geladen werden.

WICHTIG: Nach dem Filtern können keine Änderungen mehr gespeichert werden, um das Original zu schützen.
Die History wird geleert!

Args:
    addressRestriciton: Dictionary mit Adressbeschränkungen (z.B. {"thoroughfareName": "Stakenholt"})
    borderCoordinates: Liste von Koordinaten für einen rechteckigen Bereich (z.B. [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])

Returns:
    dict: Informationen zum gefilterten Dataset (building_count, minimum, maximum, filepath, title)

## `analyse_dataset`

```python
analyse_dataset() -> dict
```

Analysiert das Dataset und gibt detaillierte Informationen über die Gebäude zurück.

Returns:
    dict: Analyse-Ergebnisse mit Informationen über GML-Version, CRS, LoD, Anzahl Gebäude etc.

## `number_of_buildings`

```python
number_of_buildings() -> int
```

Gibt die Anzahl der Gebäude im aktuellen Dataset zurück.

Returns:
    int: Anzahl der Gebäude

## `get_all_buildings`

```python
get_all_buildings() -> list[dict]
```

Gibt Informationen zu allen Gebäuden im Dataset zurück.

Returns:
    list[dict]: Liste mit Dictionaries aller Gebäude und deren Attributen

## `get_buiding_Id_list`

```python
get_buiding_Id_list() -> list[str]
```

Gibt eine Liste aller Gebäude-IDs im Dataset zurück.

Returns:
    list[str]: Liste aller gml:id Werte der Gebäude

## `get_party_walls`

```python
get_party_walls() -> list
```

Findet alle angrenzenden Wände (Party Walls) zwischen Gebäuden im Dataset.

Returns:
    list: Liste aller erkannten Party Walls als [id of b0, id of w0, id of b1, id of w1, area, collision coordinates]

## `get_building_by_id`

```python
get_building_by_id(building_id: str) -> dict
```

Gibt alle Informationen zu einem spezifischen Gebäude zurück.

Args:
    building_id: Die gml:id des Gebäudes

Returns:
    dict: Dictionary mit allen Gebäude-Attributen

## `calculate_roof_volume_by_id`

```python
calculate_roof_volume_by_id(building_id: str) -> float
```

Berechnet das Dachvolumen für ein spezifisches Gebäude.

Args:
    building_id: Die gml:id des Gebäudes

Returns:
    float: Das berechnete Dachvolumen in Kubikmetern

## `create_building`

```python
create_building(id: str, groundsCoordinates: list[list[float]], groundSurfaceHeight: float, geometryHeight: float=None, roofType: str=None, roofHeight: float=None, roofOrientation: int=None, lod: int=2, isRoofEdge: bool=False) -> dict
```

Erstellt ein neues Gebäude mit wählbarem Detaillierungsgrad (LoD 0/1/2).

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

## `remove_building_from_dataset`

```python
remove_building_from_dataset(building_id: str) -> dict
```

Entfernt ein Gebäude aus dem Dataset.

Die Änderung wird direkt im Dataset vorgenommen (Single Source of Truth).
Verwende take_snapshot() vor wichtigen Änderungen und save_dataset() zum Speichern.

Args:
    building_id: Die gml:id des zu entfernenden Gebäudes

Returns:
    dict: Bestätigungsmeldung mit Anzahl verbleibender Gebäude

## `enrich_building`

```python
enrich_building(building_id: str, measured_height: float=None, roof_type: str=None, roof_height: float=None, function: str=None, usage: str=None, year_of_construction: int=None, storeys_above_ground: int=None, storeys_below_ground: int=None, creation_date: str=None) -> dict
```

Reichert ein Gebäude mit semantischen Informationen an.

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

## `remove_building_attributes`

```python
remove_building_attributes(building_id: str, attributes: list[str]) -> dict
```

Entfernt Attribute eines Gebäudes (setzt sie auf None).

Args:
    building_id: Die gml:id des Gebäudes
    attributes: Liste von Attributnamen (gleiche Namen wie in enrich_building,
        z.B. "measuredHeight", "roofType"). Unbekannte Namen werden ignoriert.

Returns:
    dict: Das aktualisierte Gebäude mit allen Attributen

## `take_snapshot`

```python
take_snapshot(description: str='') -> dict
```

Erstellt einen Snapshot des aktuellen Dataset-Zustands.

Wie ein Git-Commit: Speichert den aktuellen Zustand für spätere Rollbacks.

Args:
    description: Beschreibung des Snapshots

Returns:
    dict: Informationen über den erstellten Snapshot

## `rollback_to_snapshot`

```python
rollback_to_snapshot(snapshot_index: int) -> dict
```

Stellt einen früheren Dataset-Zustand wieder her.

Wie Git-Reset: Kehrt zu einem gespeicherten Zustand zurück.
Alle Änderungen nach diesem Snapshot gehen verloren!

Args:
    snapshot_index: Index des Snapshots (von get_dataset_history() erhalten)

Returns:
    dict: Informationen über den Rollback

## `save_dataset`

```python
save_dataset(description: str='', target_format: Optional[str]=None) -> dict
```

Speichert das Dataset dauerhaft in die GML/JSON-Datei.

Übernimmt alle aktuellen Änderungen permanent in die Datei.

Args:
    description: Beschreibung des Speicherns
    target_format: Optionales Zielformat ("gml" oder "json"). Wenn None, wird das ursprüngliche Format beibehalten.
Returns:
    dict: Informationen über den Speichervorgang

## `get_dataset_history`

```python
get_dataset_history() -> dict
```

Zeigt die Historie aller Snapshots und Speicherungen.

Returns:
    dict: Übersicht über Snapshots, Speicherungen und aktuellen Zustand
