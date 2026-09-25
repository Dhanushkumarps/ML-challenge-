# Business Entity Resolution

Entity resolution system for matching business records across multiple data sources.

## Project Structure

- `src/` - Source code and experimentation
- `output/` - Generated predictions and results
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Dataset

Training data contains:

- `train_source1.tsv`
- `train_source2.tsv`
- `train_source3.tsv`
- `train_ground_truth.tsv`

## Goal

Identify records from different sources that represent the same real-world business.

## Observed Data Noise

Initial manual inspection identified:

- Typos and spelling variations
- Abbreviations
- Case differences
- Missing values
- Partial addresses
- Reordered address components
- Accented characters
- Different writing systems/scripts
- Numeric formatting differences