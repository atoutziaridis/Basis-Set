# Task 2: Feature Engineering and Signals

## Objective
Transform raw GitHub data into meaningful signals that predict "category-defining potential" without current institutional funding.

## Input
- Enriched dataset from Task 1
- Raw text data (READMEs, descriptions, commit messages)

## Key Activities

### 2.1 Execution & Velocity Signals (Priority: High)
**Implementation**: Time series analysis + statistical measures
- **Commit Velocity**: Robust slope estimation (Theil-Sen) of weekly commits over 26 weeks
- **Release Cadence**: Median days between releases, gap penalty scoring
- **Maintenance Activity**: Issue/PR response times, recent activity indicators
- **Development Consistency**: Coefficient of variation in commit patterns

### 2.2 Team & Community Signals (Priority: High)
**Implementation**: Network analysis + statistical measures
- **Bus Factor**: 1 - (max_contributor_commits / total_commits)
- **Contributor Diversity**: Gini coefficient of commit distribution
- **Community Health**: Star velocity (normalized by age), fork-to-star ratio
- **Responsiveness**: Median time-to-first-response on issues/PRs

### 2.3 Technical Maturity Indicators (Priority: Medium)
**Implementation**: Binary scoring + weighted combinations
- **Operational Readiness**: CI/CD presence, test coverage indicators, Docker support
- **Code Quality**: Linting configs, type checking, documentation ratio
- **API Stability**: Semantic versioning, changelog presence, release notes quality

### 2.4 Market Positioning Signals (Priority: High)
**Implementation**: NLP + embedding-based analysis
- **Problem Ambition**: README sentiment analysis, market size indicators
- **Differentiation Score**: Cosine similarity to template repositories (lower = more unique)
- **Commercial Viability**: License permissiveness, API examples, deployment docs

### 2.5 Composite Score Calculation (Priority: High)
**Implementation**: Weighted combination with normalization
- Normalize all signals to [0,1] scale using robust min-max scaling
- Weight assignment based on correlation with manual "interesting" labels (if available)
- Handle missing values with domain-appropriate defaults

## Deliverables
- Feature engineering pipeline (Python scripts)
- Processed dataset with 15-20 engineered features
- Feature importance analysis and correlation matrix
- Signal validation report with distributions and outlier analysis

## Time Estimate
2-3 days

## Technical Requirements
- Python libraries: scikit-learn, numpy, pandas, scipy, nltk/spaCy
- Statistical analysis for robust estimators
- Text processing and embedding capabilities