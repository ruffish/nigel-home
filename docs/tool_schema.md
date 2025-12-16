# Tool definition schema

Tool definition files can be YAML or JSON.

## Common fields

- `name` (string, required): unique tool name
- `description` (string, required): what the tool does
- `type` (string, required): `ha_service` | `http` | `python`
- `args_schema` (object, optional): JSON Schema for tool arguments
- `rag` (object, optional): metadata for retrieval
  - `tags` (list[string], optional)
  - `examples` (list[string], optional)

## Type: `ha_service`

Required:

- `service`: string like `light.turn_on`

Optional:

- `target`: object passed as HA service target (e.g. `{entity_id: "light.kitchen"}`)
- `data`: object merged into service data

Example:

```yaml
name: lights_kitchen_on
description: Turn on the kitchen lights.
type: ha_service
service: light.turn_on
target:
  entity_id: light.kitchen
data:
  brightness_pct: 60
args_schema:
  type: object
  properties:
    brightness_pct:
      type: integer
      minimum: 1
      maximum: 100
  additionalProperties: false
```

## Type: `http`

Required:

- `method`: `GET` | `POST` | `PUT` | `PATCH` | `DELETE`
- `url`: full URL

Optional:

- `headers`: object
- `query`: object
- `json`: object
- `timeout_s`: number

## Type: `python`

Required:

- `script_path`: path inside the add-on data directory, typically `/data/tools/scripts/<file>.py`

Notes:

- Arguments are passed to stdin as JSON.
- The script should print JSON to stdout.
