# BSV Repository Prioritizer - Notebooks

This directory contains Jupyter notebooks for exploratory analysis of the BSV Repository Prioritization system.

## Available Notebooks

### exploration.ipynb (Planned)
- Comprehensive exploratory analysis of results
- Top 10 repository analysis
- Component score visualization
- Feature importance analysis
- Investment brief summaries

## Usage

To run the notebooks:

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Install Jupyter if not already installed:
   ```bash
   pip install jupyter matplotlib seaborn
   ```

3. Start Jupyter:
   ```bash
   jupyter notebook notebooks/
   ```

4. Open and run the desired notebook

## Data Sources

The notebooks analyze data from:
- `/output/bsv_prioritized_repositories.csv` - Final ranked results
- `/data/test_task3_dataset.csv` - Feature dataset
- `/data/task4_explanations.json` - Explainability analysis
- `/data/task4_evaluation_report.json` - System validation results
- `/data/task4_bias_analysis.json` - Bias detection results
