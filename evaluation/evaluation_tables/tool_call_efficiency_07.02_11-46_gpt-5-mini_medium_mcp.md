# Tool-Call Effizienz

Run: `07.02_11-46_gpt-5-mini_medium` | Setup: `mcp`

| Test Case | Runs | Avg Calls | Median Calls | Avg Overhead Calls | Order OK % | Params OK % | Both OK % | list_datasets used % | Extra-Tools Runs % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| add_building | 20 | 4.85 | 5.0 | 0.85 | 35.0 | 100.0 | 35.0 | 5.0 | 65.0 |
| highest_measured_height | 20 | 2.95 | 3.0 | 0.95 | 90.0 | 100.0 | 90.0 | 85.0 | 0.0 |
| list_buildings | 20 | 2.85 | 3.0 | 0.9 | 90.0 | 95.0 | 90.0 | 95.0 | 0.0 |
| raise_tallest_building | 20 | 6 | 6.0 | 2 | 100.0 | 95.0 | 95.0 | 30.0 | 90.0 |
| roof_volume_sum | 20 | 38.95 | 39.0 | 0.95 | 100.0 | 100.0 | 100.0 | 95.0 | 0.0 |
