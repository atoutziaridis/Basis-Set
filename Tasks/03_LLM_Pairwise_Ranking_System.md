# Task 3: LLM Pairwise Ranking System

## Objective
Implement an AI-driven pairwise comparison system using LLMs to judge repository potential, then aggregate into final rankings.

## Input
- Processed dataset with engineered features from Task 2
- Repository "cards" (structured summaries)

## Key Activities

### 3.1 Repository Card Generation (Priority: High)
**Implementation**: Template-based structured summaries
- **Card Template**: 400-500 token standardized format
  - Repository name, domain, one-liner description
  - Key metrics: stars, commits, contributors, age
  - Execution score, community score, technical maturity
  - Latest meaningful commit, license, documentation quality
- **Consistency**: Ensure uniform format to reduce LLM bias
- **Information Density**: Balance informativeness with token efficiency

### 3.2 Strategic Pair Selection (Priority: Medium)
**Implementation**: Intelligent sampling strategy
- **Initial Sampling**: Random pairs for baseline (50-100 pairs)
- **Uncertainty Sampling**: Focus on repositories with similar preliminary scores
- **Coverage Optimization**: Ensure all repositories appear in multiple comparisons
- **Target**: 200-400 total pairwise comparisons (manageable LLM cost)

### 3.3 LLM Judge Implementation (Priority: High)
**Implementation**: Structured prompting with OpenAI/Claude API
- **Prompt Design**: 
  - Clear rubric focusing on category-defining potential
  - Emphasis on execution, differentiation, market opportunity
  - Explicit instruction to downweight vanity metrics
  - Request confidence level and reasoning
- **Output Parsing**: Structured JSON response (winner, confidence, reason)
- **Quality Control**: Temperature=0 for consistency, retry on parsing failures

### 3.4 Bradley-Terry Ranking Aggregation (Priority: High)
**Implementation**: Statistical ranking model
- **Model**: Fit Bradley-Terry model on pairwise outcomes
- **Scoring**: Convert to probability-based rankings
- **Validation**: Bootstrap confidence intervals, rank stability analysis
- **Calibration**: Isotonic regression for score interpretation

### 3.5 Lightweight Distillation Model (Priority: Low)
**Implementation**: Optional XGBoost ranking model
- Train on engineered features to mimic LLM preferences
- Provides interpretable feature importance
- Faster inference for future use
- Validation against LLM rankings

## Deliverables
- Repository card generator (Python script)
- LLM judging pipeline with error handling
- Pairwise comparison dataset
- Bradley-Terry ranking model
- Final LLM-based priority scores

## Time Estimate
3-4 days

## Technical Requirements
- OpenAI/Claude API access with budget management
- Python libraries: openai, scipy, scikit-learn, choix (Bradley-Terry)
- JSON parsing and error handling
- Statistical ranking algorithms

## Cost Considerations
- Estimated 200-400 LLM API calls at ~500 tokens each
- Approximately $20-50 in API costs depending on model choice
- Implement caching to avoid re-running comparisons