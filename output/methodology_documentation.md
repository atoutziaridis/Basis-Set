# BSV Repository Prioritization - Methodology Documentation
**Version**: 1.0
**Date**: September 14, 2025

## Overview
This document describes the methodology used to prioritize GitHub repositories 
for Basis Set Ventures' investment analysis. The system combines multiple data 
sources and AI-powered analysis to identify category-defining companies without 
current institutional funding.

## Data Collection (Task 1)
### GitHub API Integration
- Repository metadata: stars, forks, creation date, language distribution
- Activity metrics: commit frequency, release cadence, issue response times
- Contributor analysis: bus factor, contribution distribution, team diversity
- Code quality indicators: CI/CD presence, test coverage, documentation quality

### Funding Detection
- NLP-based text analysis of README files and repository descriptions
- Pattern matching for funding keywords and investor mentions
- Risk classification: low_risk_unfunded, unfunded, funded, high_risk_funded

## Feature Engineering (Task 2)
### Composite Scores
- **Execution Velocity**: Commit patterns, release cadence, development consistency
- **Team Resilience**: Contributor diversity, bus factor, community health
- **Technical Maturity**: Code quality, documentation, operational readiness
- **Market Positioning**: Commercial viability, technology differentiation

## LLM Pairwise Ranking (Task 3)
### Repository Cards
- Structured summaries highlighting key features and innovations
- Technology stack, use cases, and differentiation factors
- Market potential and adoption indicators

### Pairwise Comparisons
- GPT-4 powered comparative analysis
- Bradley-Terry model for consistent ranking from pairwise judgments
- Confidence intervals and stability analysis

## Final Scoring Framework (Task 4)
### Weighted Linear Combination
```
Final Score = (
    0.60 × LLM Preference Score +
    0.15 × Technical Execution +
    0.15 × Market Adoption +
    0.10 × Team Resilience
) × Funding Gate Multiplier
```

### Funding Gate
- Multiplier: max(0.6, 1 - p_institutional_funding)
- Ensures preference for unfunded projects while not completely excluding funded ones
- Based on funding confidence score from text analysis

## System Validation
### Ablation Studies
- LLM-only rankings: Tests pure AI judgment effectiveness
- Features-only rankings: Validates traditional metrics approach
- Component removal: Measures individual component contributions
- Equal weights: Compares to optimized weighting scheme

### Sanity Checks
- Star count correlation: Should be moderate (not too high/low)
- Age bias: Should not favor old or new repositories excessively
- Activity correlation: Should positively correlate with development activity
- Funding bias: Should not favor funded projects

### Bias Detection
- **Age Bias**: Repository age correlation analysis
- **Popularity Bias**: Over-dependence on star count
- **Language Bias**: Programming language preferences
- **Size Bias**: Repository size metric correlations
- **Temporal Bias**: Recent activity preferences
- **Funding Bias**: Unintended funding status preferences

## Limitations and Future Work
### Current Limitations
- Limited to public GitHub data
- Text-based funding detection may have false negatives
- Market potential based on technical signals, not market research
- Small sample size for LLM comparisons due to API costs

### Future Improvements
- Integration with additional data sources (Crunchbase, AngelList)
- Enhanced funding detection using structured data
- Market size and opportunity assessment
- Real-time monitoring and alert systems
- Expanded LLM comparison coverage