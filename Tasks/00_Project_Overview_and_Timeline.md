# Project Overview and Timeline

## Project Goal
Rank 100 GitHub repositories by their potential to become category-defining companies without current institutional funding, using modern AI approaches and comprehensive data analysis.

## Core Requirements Met
- ✅ **Modern AI Focus**: LLM pairwise judging as primary ranking signal
- ✅ **Data Enrichment**: Beyond basic metrics to capture execution, adoption, and innovation
- ✅ **Explainable Results**: Clear reasoning codes for each ranking decision
- ✅ **Funding Filter**: Detect and downweight institutionally-funded projects
- ✅ **Reproducible Pipeline**: Single-command execution with comprehensive documentation

## Key Simplifications Made
1. **Reduced Complexity**: Focused on most impactful signals rather than all possible features
2. **Practical Scope**: 200-400 LLM comparisons vs 4,950 possible pairs (cost/time efficient)
3. **Streamlined Architecture**: Direct implementation vs full research framework
4. **Focused Evaluation**: Core ablations vs exhaustive experimentation

## Task Dependencies
```
Task 1 (Data Collection) 
    ↓
Task 2 (Feature Engineering)
    ↓
Task 3 (LLM Ranking) ← Can start after Task 2 is 50% complete
    ↓
Task 4 (Final Scoring) ← Needs Tasks 2 & 3 complete
    ↓  
Task 5 (Implementation) ← Integration of all tasks
```

## Estimated Timeline
- **Total Duration**: 12-15 days
- **Parallel Work Opportunities**: Data collection can overlap with feature engineering setup
- **Critical Path**: LLM ranking system (highest complexity and API dependencies)

## Task Breakdown
| Task | Duration | Priority | Dependencies |
|------|----------|----------|--------------|
| 1. Data Collection | 2-3 days | High | None |
| 2. Feature Engineering | 2-3 days | High | Task 1 |
| 3. LLM Ranking | 3-4 days | Critical | Task 2 (partial) |
| 4. Final Scoring | 2-3 days | High | Tasks 2 & 3 |
| 5. Implementation | 2-3 days | High | All tasks |

## Success Metrics
1. **Output Quality**: Final CSV with interpretable rankings that surface innovative, unfunded projects
2. **Methodology Rigor**: Clear evaluation showing the approach works better than simple heuristics  
3. **Reproducibility**: BSV team can re-run analysis with new data
4. **Innovation**: Demonstrates modern AI techniques in practical investment context

## Risk Mitigation Strategies
- **API Rate Limits**: Implement caching and respect GitHub API constraints
- **LLM Costs**: Smart pair selection to minimize API calls while maximizing ranking quality
- **Time Management**: Core functionality first, enhancements only if time permits
- **Data Quality**: Robust error handling for missing or malformed repository data

## Expected Outcomes
- Ranked list of 100 repositories with confidence scores
- Clear methodology that BSV can apply to future deal sourcing
- Identification of 10-20 high-potential, unfunded projects for investment consideration
- Demonstrable improvement over naive popularity-based ranking