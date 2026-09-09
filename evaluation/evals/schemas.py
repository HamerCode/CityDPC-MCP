RESPONSE_SCHEMAS = {
    "list_buildings": {
        "type": "json_schema",
        "name": "building_list_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "building_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Complete list of all building gml:ids found in the dataset"
                },
                "total": {
                    "type": "integer",
                    "description": "Total number of buildings"
                }
            },
            "required": ["building_ids", "total"],
            "additionalProperties": False
        }
    },
    "highest_measured_height": {
        "type": "json_schema",
        "name": "highest_building_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "building_id": {
                    "type": "string",
                    "description": "The gml:id of the building with the highest measuredHeight"
                },
                "measuredHeight": {
                    "type": "number",
                    "description": "The measured height value in meters"
                },
                "missing_count": {
                    "type": "integer",
                    "description": "Number of buildings without measuredHeight attribute"
                }
            },
            "required": ["building_id", "measuredHeight", "missing_count"],
            "additionalProperties": False
        }
    },
    "count_party_walls": {
        "type": "json_schema",
        "name": "party_walls_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "party_wall_count": {
                    "type": "integer",
                    "description": "Number of party walls (adjacent wall surfaces between neighbouring buildings) in the dataset"
                }
            },
            "required": ["party_wall_count"],
            "additionalProperties": False
        }
    },
    "raise_building_by_id": None,
    "add_building": None,
    "ai_campus": None,
    "rwth_hauptgebaeude": None,
    "rwth_hauptgebaeude_groundplan": None,
}
