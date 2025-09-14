# Task 1: Data Collection and Enrichment

## Objective
Enrich the provided dataset with comprehensive GitHub metrics and external data sources to create a robust foundation for ranking analysis.

## Input
- CSV file with 100 GitHub repositories (name, description, basic metrics)
- GitHub URLs for each repository

## Key Activities

### 1.1 GitHub API Data Collection (Priority: High)
**Implementation**: Python script using PyGithub or requests
- **Repository Metadata**: stars, forks, watchers, creation date, last push, language distribution
- **Activity Metrics**: commit frequency (last 6 months), issue/PR open/close rates, release cadence
- **Contributors**: contributor count, top contributors, commit distribution (bus factor calculation)
- **Dependencies**: extract from package.json, requirements.txt, Cargo.toml, etc.

### 1.2 Code Quality Indicators (Priority: Medium)
**Implementation**: Shallow git clone + file analysis
- **Development Practices**: presence of CI/CD (.github/workflows), tests directory, linting configs
- **Documentation**: README quality score (length, sections, examples), docs folder presence
- **Code Structure**: file organization, configuration files (Docker, package managers)

### 1.3 Adoption Signals (Priority: High)
**Implementation**: Web scraping + API calls
- **Package Downloads**: PyPI, npm, crates.io download statistics (where applicable)
- **Dependents**: GitHub dependents page scraping (lightweight)
- **Community**: GitHub stars growth rate, fork-to-star ratio

### 1.4 Funding Detection (Priority: High)
**Implementation**: NLP pattern matching + manual validation
- **Text Analysis**: Scan README, website, GitHub organization for funding keywords
- **Pattern Matching**: "Series A", "seed round", "backed by", "venture", company domains
- **Manual Validation**: Flag uncertain cases for human review

## Deliverables
- Enriched dataset (CSV) with 25-30 additional features per repository
- Data collection scripts with rate limiting and caching
- Data quality report highlighting missing values and collection issues

## Time Estimate
2-3 days

## Technical Requirements
- GitHub API token (5000 requests/hour limit consideration)
- Python libraries: PyGithub, requests, beautifulsoup4, pandas
- Rate limiting and error handling implementation