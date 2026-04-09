# PropEdge V16.0 — Complete Package

Everything is pre-placed. Just run:

```bash
cd PropEdgeV16
python3 run.py setup     # builds season JSONs + trains Elite V2 GBT (~10 min)
python3 run.py install   # install launchd scheduler
python3 run.py status    # verify
```

## Batch Schedule (UK time)
| Batch | Time  | Action |
|-------|-------|--------|
| B0    | 07:00 | Grade yesterday + monthly retrain |
| B1    | 08:30 | Morning scan |
| B2    | 11:00 | Mid-morning refresh |
| B3    | 16:00 | Afternoon sweep |
| B4    | 18:30 | Pre-game final |
| B5    | 21:00 | Late West Coast |

## Elite Tier System
| Tier   | Threshold | Stake |
|--------|-----------|-------|
| APEX   | ≥0.81     | 2.5u  |
| ULTRA  | ≥0.78     | 2.0u  |
| ELITE  | ≥0.75     | 2.0u  |
| STRONG | ≥0.72     | 1.5u  |
| PLAY+  | ≥0.68     | 1.2u  |

## What's included
- **source-files/**: Both game log CSVs, H2H database, Props Excel
- **models/V9.2–V14/**: All pkl files from your uploaded Model.zip
- **models/elite/**: Empty — populated by `python3 run.py retrain`
- **data/**: Empty — populated by setup and daily batches
