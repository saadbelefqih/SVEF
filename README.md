# SVEF: Schema Validation and Evaluation Framework

This repository contains the reference implementation of **SVEF (Schema Validation and Evaluation Framework)**, a framework for evaluating schemas extracted from **JSON-based schemaless databases**.

SVEF evaluates extracted schemas across six dimensions:

- **Data Type Accuracy (DTA)**
- **Required and Optional Fields**
- **Multiple Type Support (MTS)**
- **Collection Structure Consistency (CSC)**
- **Entity Relationship Recovery (ERR)**
- **Temporal Evolution Detection (TED)**

The implementation supports evaluation against **reference schemas** and benchmark datasets, and is associated with the paper:

**Schema Validation and Evaluation Framework: Assessing Extracted Schemas in JSON-Based Databases**

## Repository Structure

```text
SVEF/
├── README.md
├── SVEF.py
├── dataset.py
schemas/
├── ecommerce_ground_truth.json
└── ecommerce_inferred.json
```

## Schema Representation

In this repository, the term **schema** does not refer only to a standard JSON Schema document.  
Instead, it denotes a structured JSON-based representation used for evaluation in SVEF, including:

- properties and types
- required and optional fields
- heterogeneous typing
- array structures
- inter-entity relationships
- temporal variants, when relevant

## Requirements

- Python 3.10+
- pip

Install dependencies with:

```bash
pip install -r 
```

## Installation

```bash
git clone https://github.com/saadbelefqih/SVEF.git
cd SVEF
pip install -r requirements.txt
```

## Running the Evaluation

Basic execution:

```bash
python SVEF.py
```

Example execution:

```bash
python SVEF.py \
  --dataset datasets/ecommerce/orders.json \
  --reference schemas/ecommerce_ground_truth.json \
  --inferred schemas/ecommerce_inferred.json \
  --output results/ecommerce_results.json
```

## Inputs

SVEF expects:

1. **Dataset records** in JSON or JSON-like form  
2. **Schema artefacts** in JSON format:
   - reference / ground-truth schemas
   - inferred schemas

## Outputs

The evaluation produces:

- dimension-level scores
- overall **Schema Quality Score (SQS)**
- result files in JSON format

Example output:

```json
{
  "dataset": "ecommerce",
  "scores": {
    "DTA": 0.94,
    "PresenceAccuracy": 0.88,
    "MTS": 0.91,
    "CSC": 0.86,
    "ERR": 0.84,
    "TED": 0.79
  },
  "SQS": 0.87
}
```

## Reproducing the Paper Results

To reproduce the experiments:

1. Install the dependencies
2. Use the datasets in `datasets/`
3. Use the corresponding reference schemas in `schemas/`
4. Run `SVEF.py`
5. Compare the generated outputs in `results/` with the results reported in the paper

## Notes

The repository uses a JSON-based schema representation designed for SVEF evaluation.  
This representation is broader than standard JSON Schema because the framework also evaluates relationships and temporal schema evolution.


## Contact

For questions or issues, please open an issue in the repository.
