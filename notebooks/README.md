# BSV Repository Prioritizer - Analysis Notebooks

This directory is reserved for Jupyter notebooks to explore and visualize the BSV Repository Prioritization system results.

## Potential Analysis Notebooks

### 1. Results Exploration
- Top 10 BSV investment targets analysis
- Score distribution and variance analysis  
- Technical execution component breakdown
- AI/ML bias system effectiveness

### 2. Methodology Validation
- Feature importance analysis across 5 technical components
- LLM preference score vs final ranking correlation
- BSV investment score methodology validation
- Time series momentum analysis visualization

### 3. Investment Intelligence
- Category-defining potential assessment
- Funding advantage analysis for unfunded projects
- Market timing and technical readiness correlation
- Portfolio fit analysis for BSV investment thesis

## Usage

To create and run analysis notebooks:

1. **Activate Environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install Jupyter**:
   ```bash
   pip install jupyter matplotlib seaborn plotly
   ```

3. **Start Jupyter**:
   ```bash
   jupyter notebook notebooks/
   ```

4. **Create New Notebook** and analyze results from:
   - `output/bsv_enhanced_final_rankings.csv` - Final enhanced rankings
   - `data/final_engineered_features.csv` - 63 engineered features
   - `data/complete_real_dataset_llm_rankings.csv` - Real LLM preference scores

## Sample Analysis Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load enhanced results
df = pd.read_csv('../output/bsv_enhanced_final_rankings.csv')

# Analyze score distributions
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(df['technical_execution_score'], bins=20, alpha=0.7)
plt.title('Technical Execution Score Distribution')

plt.subplot(2, 2, 2)
plt.hist(df['bsv_investment_score'], bins=20, alpha=0.7)
plt.title('BSV Investment Score Distribution')

plt.subplot(2, 2, 3)
plt.hist(df['innovation_score'], bins=20, alpha=0.7)
plt.title('Innovation Score Distribution')

plt.subplot(2, 2, 4)
plt.scatter(df['technical_execution_score'], df['final_score'])
plt.title('Technical vs Final Score Correlation')

plt.tight_layout()
plt.show()

# Top 10 analysis
top10 = df.head(10)
print("Top 10 BSV Investment Targets:")
print(top10[['repo_name', 'final_score', 'technical_execution_score', 'bsv_investment_score']])
```

## Data Sources

The analysis notebooks can explore:
- **Enhanced Rankings**: `output/bsv_enhanced_final_rankings.csv`
- **Engineered Features**: `data/final_engineered_features.csv` 
- **LLM Analysis**: `data/complete_real_dataset_llm_rankings.csv`
- **Methodology Docs**: `BSV_Methodology_Summary.md`