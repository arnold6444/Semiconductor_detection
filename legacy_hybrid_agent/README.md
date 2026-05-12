# Legacy Hybrid Agent

This folder preserves the earlier `hy_ngv` implementation after consolidating the portfolio project into `Semiconductor_detection`.

## Role in this repository

- Main implementation: root `main.py`, `config.py`, and `src/`
- Legacy reference: `legacy_hybrid_agent/agent/`

The legacy version used a local model first, then sent ambiguous samples to an external LLM decision-support step.

```text
Local Model -> Ambiguity Check -> External LLM if ambiguous -> Decision Fusion
```

## What was intentionally not copied

- trained model weights (`model.pth`)
- cached images
- `__pycache__` files
- generated submission/output files

Those files are runtime artifacts, not source code. Recreate them locally when needed.

## Run Legacy Version

From this folder:

```bash
pip install -r requirements.txt
set LUXIA_API_KEY=your_api_key_here
python -m agent.run --train --train_csv train.csv
python -m agent.run --input_csv eval.csv --output submission.csv
```

If you continue this project, prefer updating the main implementation first and only use this folder as historical reference.
