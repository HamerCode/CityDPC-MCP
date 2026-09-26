"""The five benchmark tasks of the paper (prompts are used verbatim)."""

TEST_BUILDING_SPEC = {
    "id": "EVAL_TEST_BUILDING_001",
    "groundCoordinates": [
        [292200.0, 5630700.0],
        [292210.0, 5630700.0],
        [292210.0, 5630712.0],
        [292200.0, 5630712.0],
        [292200.0, 5630700.0]
    ],
    "groundSurfaceHeight": 198.5,
    "measuredHeight": 10.5,
    "storeysAboveGround": 3,
    "roofType": "1030",
    "roofHeight": 3.0,
    "function": "31001_1010",
}

RAISE_BUILDING_SPEC = {
    "target_building_id": "DENW39AL100023tX",
    "old_height": 1.553,
    "increase_by": 2.0,
    "new_height": 3.553
}

TEST_CASES = [
    {
        "id": "list_buildings",
        "category": "query",
        "prompt": (
            "Bei dem Dataset 'evaluation.{format_type}' bitte alle "
            "Gebäude-IDs (gml:id) auflisten und am Ende die Gesamtanzahl angeben."
        ),
    },
    {
        "id": "highest_measured_height",
        "category": "query",
        "prompt": (
            "Bei dem Dataset 'evaluation.{format_type}' bitte für alle "
            "Gebäude measuredHeight prüfen und die Building-ID des Gebäudes mit der "
            "größten measuredHeight sowie die zugehörige Höhe ausgeben."
        ),
    },
    {
        "id": "count_party_walls",
        "category": "analytic",
        "prompt": (
            "Im Dataset 'evaluation.{format_type}': Wie viele Party Walls "
            "(angrenzende Wandflächen zwischen benachbarten Gebäuden) gibt es? "
            "Gib nur die Anzahl als ganze Zahl aus."
        ),
    },
    {
        "id": "raise_building_by_id",
        "category": "state_change",
        "prompt": (
            f"Bei dem Dataset 'evaluation.{{format_type}}': Finde das Gebäude mit der ID "
            f"'{RAISE_BUILDING_SPEC['target_building_id']}' und erhöhe dessen measuredHeight "
            f"um {RAISE_BUILDING_SPEC['increase_by']}m. "
            "Führe die Änderung tatsächlich durch und bestätige am Ende die Building-ID, "
            "die alte Höhe und die neue Höhe."
        ),
    },
    {
        "id": "add_building",
        "category": "state_change",
        "prompt": (
            f"Bei dem Dataset 'evaluation.{{format_type}}': Füge ein neues Gebäude mit folgenden Spezifikationen hinzu:\n"
            f"- Building-ID: '{TEST_BUILDING_SPEC['id']}'\n"
            f"- Grundriss (2D-Koordinaten, geschlossen): {TEST_BUILDING_SPEC['groundCoordinates']}\n"
            f"- Bodenhöhe: {TEST_BUILDING_SPEC['groundSurfaceHeight']}m, "
            f"measuredHeight: {TEST_BUILDING_SPEC['measuredHeight']}m, "
            f"Dachhöhe: {TEST_BUILDING_SPEC['roofHeight']}m\n"
            f"- Stockwerke über Grund: {TEST_BUILDING_SPEC['storeysAboveGround']}, "
            f"Dachtyp: '{TEST_BUILDING_SPEC['roofType']}', "
            f"Funktion: '{TEST_BUILDING_SPEC['function']}'\n\n"
            "Führe die Erstellung tatsächlich durch und bestätige am Ende "
            "Building-ID, measuredHeight, Stockwerke, Dachtyp und Funktion."
        ),
    },
]


def get_test_cases(format_type="city.json", test_ids: list[str] | None = None):
    out = []
    for tc in TEST_CASES:
        if test_ids is not None and tc["id"] not in test_ids:
            continue
        tc = tc.copy()
        tc["prompt"] = tc["prompt"].replace("{format_type}", format_type)
        out.append(tc)
    return out


BASE_INSTRUCTIONS = """Du bist ein hilfreicher CityJSON-Experte, der Fragen beantworten und bei Aufgaben helfen kann.

WICHTIG:
1. Stelle KEINE Rückfragen — antworte direkt und vollständig.
2. Gib konkrete Zahlen und Ergebnisse aus, keine theoretischen Erklärungen.
"""


SETUP_SPECIFIC_INSTRUCTIONS = {
    "no_tools": (
        "Setup: Du hast keine Tools. Das Dataset ist unten an den Prompt angehängt. "
        "Für State-Changes gib einen JSON-Patch nach RFC 6902 aus."
    ),
    "mcp": (
        "Setup: Du hast MCP-Tools für CityJSON-Manipulation verfügbar — nutze sie. "
        "Bei State-Changes (Modifikationen) musst du das Dataset am Ende speichern, "
        "damit die Änderung persistiert wird."
    ),
    "code_interpreter": (
        "Setup: Das Dataset ist unten an den Prompt angehängt und liegt zusätzlich "
        "als Datei auf dem Host — der absolute Pfad steht unten ('Dataset-Pfad:'). "
        "Du hast eine Python-Sandbox als optionales Hilfsmittel — nutze sie nur, "
        "wenn sie sich für die Aufgabe wirklich eignet. Wenn du sie nutzt: "
        "kopiere die Datei mit copy_file(container_id, dest_path, local_src_file=<pfad>) "
        "in die Sandbox, statt den Inhalt selbst abzutippen. "
        "Bei State-Changes in der Sandbox musst du die modifizierte Datei am Ende mit "
        "copy_file_from_sandbox(container_id, container_src_path, local_dest_path=<pfad>) "
        "zurück auf den Host kopieren, sonst geht die Änderung verloren. "
        "Für State-Changes kannst du alternativ auch einen JSON-Patch nach RFC 6902 "
        "ausgeben — der wird dann lokal auf das Dataset angewendet."
    ),
}


def get_setup_specific_prompt(setup_key: str, test_case_id: str | None = None) -> str:
    """Liefert den kurzen Setup-Hinweis für ein Setup."""
    return SETUP_SPECIFIC_INSTRUCTIONS.get(setup_key, "").strip()
