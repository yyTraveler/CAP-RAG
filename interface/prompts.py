preparse_prompt_schema = {
    "type": "json_schema",
    "properties": {
        "query": {
            "type": "string",
            "description": "propose a possiable question which is deep and could get the answer from the content field."
        },
        "summary": {
            "type": "string",
            "description": "a detailed description of this context. you should make sure this description can greatly help reader undestand the original text."
        },
        "points": {
            "type": "array",
            "items": {
                "$ref": "#/$defs/points"
            },
            "description": "a list of main points of the text"
        }
    },
    "$defs": {
        "points": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "propose a possiable question which is deep and could get the answer from the content field."
                },
                "summary": {
                    "type": "string",
                    "description": "the main idea of this point or the standard answer to the query."
                },
                "reference": {
                    "type": "string",
                    "description": "the original text for this point."
                }
            },
            "required": [
                "query",
                "summary"
                "reference"
            ],
            "additionalProperties": False
        }
    },
    "additionalProperties": False,
    "required": ["query", "summary", "points"]
}