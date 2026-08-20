# Structured LLM Peer Review Experiments

This repository contains the experimental materials for a dissertation investigating whether schema-constrained reviewing reduces two failures of LLM peer review: susceptibility to document-layer prompt injection and weak responses to faulty research logic. It also compares review-aspect coverage, generation efficiency, and rating variation between Free and Structured reviewing.

## Repository structure

- `llm_peer_review_experiment.ipynb`: main review-generation and metric-extraction pipeline.
- `data/`: source metadata and controlled experimental inputs.
- `outputs/`: generated reviews, extracted metrics, author annotations, summary statistics, and figures.
- `tools/`: scripts for data preparation, PDF injection, validation, analysis, and plotting.

## Setup

Use Python 3.10 or later. The main dependencies are `openai`, `pydantic`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`, `python-dotenv`, `PyMuPDF`, `requests`, `tiktoken`, and `Pillow`.

Run the main notebook from the repository root so that its relative `data/` and `outputs/` paths resolve correctly. Supporting scripts should also be run from the root, for example:

```bash
python tools/inject_pdfs.py
python tools/plot_results.py
```

## Reproducing the workflow

The notebook records the main sequence from controlled inputs to review generation and outcome extraction. Scripts in `tools/` prepare the PDFs, collect supporting evidence, reproduce the statistical summaries, and generate the dissertation figures. Paper, condition, and reviewing-setup identifiers link records across these stages.

Generated outputs are retained to make the reported analyses auditable without repeating paid API calls. API responses may vary if the workflow is rerun with different model versions or service settings.
