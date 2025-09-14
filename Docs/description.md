Blueprint: "Founder--Code Signal Graph (FCSG) + LLM Pairwise Judge"
==================================================================
Note: You don't have to follow this exactly. This serves as an idea for how to go about this project, but if something is not feasible or will take too long, feel free to make your own changes.

Goal (made precise)
-------------------

Rank 100 GitHub repos by **pre-institutional, category-defining potential**, not by current popularity. Output: CSV with rank + interpretable reasons per repo.

* * * * *

1) Data Enrichment (multi-view, all programmatic)
=================================================

**Inputs**: Airtable list of GitHub URLs.

**Collectors (Python scripts)**

-   **GitHub REST/GraphQL** (tokened):\
    Repo: name, description, topics, license, stars, forks, watchers, created_at, pushed_at, releases, issues/PRs (open/closed/merged), default_branch.\
    Contributors: logins, commits, first/last contribution.\
    Dependents (via dependency graph API if available) and **reverse dependencies** (fallback: scrape dependents page light-touch).\
    Commit history: timestamped commits, author.

-   **Code snapshot** (git clone shallow): file tree, languages (via Linguist), presence of `tests/`, `Dockerfile`, CI configs (`.github/workflows/*`), `pyproject.toml`/`package.json`, `setup.cfg`, `ruff.toml`, `mypy.ini`.

-   **Text**: README.md, docs/, CHANGELOG, commit messages (last N=300), issue titles/bodies (last N=200), PR titles (last N=200).

-   **Light external** (optional, time-boxed): PyPI/npm downloads if package exists (JSON endpoints), HuggingFace model/card existence, package dependents.

**Why this matters**: captures **execution velocity**, **developer quality**, **adoption**, and **product narrative** without hand-wavy guesswork.

* * * * *

2) Signals (novel, codable, interpretable)
==========================================

### A. Execution & Cadence

-   **Commit Velocity (robust)**: Theil--Sen slope of weekly commits over last 26 weeks.

-   **Burstiness**: Fano factor of weekly commits (variance/mean); penalize extreme sporadic bursts unless aligned with releases.

-   **Release Regularity**: median days between releases; penalty for long gaps.

-   **Maintainer Responsiveness**: median time-to-first-response on issues/PRs.

### B. Team Resilience

-   **Bus Factor**: `1 -- (max_commits_by_single_author / total_commits)`.

-   **Contributor Retention**: share of contributors active in last 60/120 days.

-   **Contributor Graph Centrality**: build a local graph where nodes are contributors; edges if they co-contributed to ≥2 repos; compute degree and eigenvector centrality using only this cohort + their known public repos (downloadable).

### C. Adoption & Pull

-   **Dependents Count**: number of repos/packages depending on this project.

-   **Normalized Star Velocity**: star slope / repo age (months), winsorized to cap viral noise.

-   **Package Downloads**: log(1 + weekly downloads), smoothed EMA(α=0.3).

### D. Product Maturity Proxies

-   **Operational Readiness**: presence of `Dockerfile`, CI, tests, typed code (mypy configs), docs coverage ratio (#doc files / KLOC).

-   **API Stability**: CHANGELOG present + semver tags.

### E. Originality vs Boilerplate

-   **Code Embedding Novelty**: embed code samples (AST-aware or code LLM embedding) → distance from a baseline of template repos (starter kits, forks) using MMD (Maximum Mean Discrepancy). High distance = **less boilerplate**.

-   **README Semantic Uniqueness**: embed README; compute distance from cluster of generic templates ("awesome-lists", cookiecutter).

-   **Duplication Check**: near-duplicate detection via MinHash on file shingles; down-weight heavy clones.

### F. Market & Vision (LLM-read)

-   **Problem Ambition** (0--1): LLM scores README/docs for clarity of problem, market size proxies (enterprise workflow, infra leverage), ecosystem fit.

-   **Differentiation** (0--1): LLM contrasts against nearest neighbors (cosine in embedding space) and judges novelty of approach.

-   **Go-to-Market Plausibility** (0--1): presence of CLI/API examples, integration points, deployment story.

-   **Safety/License Enablement** (0--1): permissive license? (MIT/Apache/BSD get a boost for adoption potential).

### G. "No Institutional Funding" Likelihood

-   **Weak-supervision**: pattern match strings in README/site ("Seed round", "Series A", "led by ...").

-   **Entity Resolver (LLM)**: extract org names, founders from README/owner profile; LLM says **Likely Funded / Likely Unfunded / Unclear** with rationale.

-   **Penalty**: multiply final score by `1 -- p(institutional)` with floor at 0.6 to avoid over-punishing uncertain cases.

* * * * *

3) Modern AI Core: LLM Pairwise Judge → Bradley--Terry Ranking
=============================================================

**Idea**: Let an LLM act as a **preference oracle** on *summaries* of each repo (not raw code), then distill those preferences into a numeric ranking. This is **learned-to-rank** with synthetic yet rubric-controlled labels.

### 3.1 Canonical "Repo Card" (auto-generated)

For each repo, generate a 400--600 token card from structured signals + snippets:

-   One-liner, domain tag(s), short bullets on Execution (velocity, releases), Adoption (dependents, packages), Team Resilience (bus factor), Originality (MMD score), Maturity (CI/Tests), License, Latest meaningful commit message.

-   Keep it **consistent schema** across repos to reduce judge bias.

### 3.2 Pairwise Judging (LLM)

-   Sample K=300--600 **informative pairs** using uncertainty sampling (start random, then focus near current score boundary).

-   Prompt (concise rubric):

    -   "Which repo is **more likely** to become a **category-defining company** *without current institutional funding*? Weigh: Ambition, Differentiation, Adoption Pull, Execution Velocity, Team Resilience, Commercializability (license/deployability). Down-weight vanity stars. Return: winner, confidence ∈ {low,med,high}, 1-sentence reason."

-   Parse JSON output; store pairwise outcomes.

### 3.3 Rank Aggregation

-   Fit a **Bradley--Terry** or **Plackett--Luce** model to pairwise outcomes → latent **LLM-Preference Score** per repo.

-   Calibrate to [0,1] with isotonic regression using soft anchors (e.g., repos with many dependents should rarely be bottom-decile).

### 3.4 Distillation to Lightweight Model (optional but recommended)

-   Train a **RankNet / XGBoost-LTR** on our handcrafted features to **mimic** the LLM judge (teacher-student).

-   Benefit: speed, determinism, feature importances.

* * * * *

4) Final Scoring (transparent)
==============================

Compute:

`CategoryPotential = 0.55 * LLM_Pref +
                    0.15 * OriginalityScore +
                    0.10 * AdoptionScore +
                    0.10 * ExecutionScore +
                    0.10 * TeamResilience
FundingGate = max(0.6, 1 - p_institutional_funding)
FinalScore = CategoryPotential * FundingGate`

Return per-repo **reason codes**: top 3 contributing features with SHAP or simple feature deltas (e.g., "High originality (MMD 0.82), strong dependents (124), steady releases (median 21d)").

* * * * *

5) Evaluation (show your work, not just a number)
=================================================

-   **Ablations**:

    -   Heuristics only vs LLM_Pref only vs combined.

    -   Remove Originality, observe rank drift.

    -   Replace Theil--Sen with OLS slope to show robustness gain.

-   **Sanity Panels**:

    -   Top 10 repos: manual spot-check with concise investment blurbs.

    -   **Leaders by single signal** (e.g., highest dependents but mediocre originality)---does the model appropriately rank/penalize?

-   **Stability**:

    -   Bootstrap input pairs and recompute Bradley--Terry → rank variance; highlight stable top-quartile.

-   **Bias checks**:

    -   Correlate FinalScore with repo age and star count; ensure **weak** correlation (we're not popularity ranking).

* * * * *

6) Implementation Plan (feasible, code-ready)
=============================================

**Repo structure**

`bsv-prioritizer/
  README.md
  env.yml / requirements.txt
  src/
    collect/
      fetch_github.py
      snapshot_code.py
      enrich_packages.py
    features/
      signals_time.py
      signals_code.py
      signals_text.py
      originality_mmd.py
    llm/
      make_repo_cards.py
      pairwise_judge.py
      bt_rank.py
      distill_ranknet.py
    modeling/
      combine_scores.py
      explain_shap.py
    utils/
      cache.py, io.py, rate_limit.py
  notebooks/
    01_collect_inspect.ipynb
    02_feature_build.ipynb
    03_llm_pairs_rank.ipynb
    04_eval_ablation.ipynb
  data/
    raw/, interim/, processed/
  output/
    prioritized.csv
    repo_cards.jsonl`

**Key techniques & libs**

-   GitHub API (PyGithub/requests + GraphQL).

-   Embeddings:

    -   **Text**: any sentence-transformers model;

    -   **Code**: a code-embedding model (e.g., small CodeT5/StarCoder embeddings);

    -   Fallback if offline: bag-of-identifiers + doc2vec (still workable).

-   MMD originality: sklearn pairwise distances + kernel two-sample test.

-   Time series: statsmodels (Theil--Sen), numpy.

-   LTR: `scikit-learn` (isotonic), `xgboost` ranking or lightgbm-ranker.

-   Pairwise ranking: `choix` (Bradley--Terry), or custom logistic.

**LLM usage**

-   Keep **token budget small**: judge from compact repo cards (no raw code).

-   Determinism: fix temperature=0; store prompts & seeds; cache outputs.

-   If APIs are constrained, swap to an open local model for judging with the same schema.

**Repro**

-   `make setup && make run` (or a bash script) that:

    1.  pulls data, 2) builds features, 3) creates repo cards, 4) judges pairs, 5) aggregates rank, 6) writes `output/prioritized.csv`.

**CSV schema**

`repo,rank,final_score,category_potential,funding_gate,
originality,adoption,execution,team_resilience,
p_institutional,reason_1,reason_2,reason_3`

* * * * *

7) Intellectual pressure-test (assumptions & counterarguments)
==============================================================

**Assumption**: GitHub signals predict "category-defining" outcomes.

-   *Skeptic*: Many iconic companies are closed-source early or keep core IP private.

-   *Mitigation*: Our signals emphasize **team execution** and **ecosystem pull**, which leak through even with partial OSS (e.g., wrappers, SDKs). Also, we down-weight raw stars and add dependents & packaging signals.

**Assumption**: LLM judgments are reliable.

-   *Skeptic*: Hallucinations, rubric drift, bias to eloquent READMEs.

-   *Mitigation*: Constrain to **structured cards**, force binary choice, aggregate over many pairs, **distill** to a numeric model, and run ablations. Add readability-bias penalty (e.g., detect excessive marketing language).

**Assumption**: Originality (distance from boilerplate) correlates with outlier success.

-   *Skeptic*: Being different ≠ valuable; many great infra tools re-implement standards.

-   *Mitigation*: Originality is only 15%; Adoption & Execution balance it. Also inspect cases where originality is high but adoption is nil.

**Assumption**: "No institutional funding" can be detected.

-   *Skeptic*: Incomplete public data.

-   *Mitigation*: Treat as probability; never hard-exclude. Keep a minimum FundingGate of 0.6 to avoid missing truly unfunded gems.

* * * * *

8) Stretch Experiments (optional if time allows)
================================================

-   **Active Learning on Pairs**: pick next pairs by **max entropy** under the Bradley--Terry posterior.

-   **Neighborhood Contrast**: for each repo, select k nearest neighbors in embedding space, have LLM judge the focal repo vs its neighbors; aggregate "beats-neighbors" rate.

-   **Graph Features++**: author-level embeddings from their prior repos (domain specialization, consistency).

-   **Timeseries Foundation**: replace simple slopes with a tiny seq model (e.g., GRU) over weekly stars/commits to detect consistent *acceleration*, not just level.

* * * * *

9) What you deliver (to match BSV's ask)
========================================

-   **Code**: the full pipeline; cached API responses; seed fixed.

-   **Output dataset**: `prioritized.csv` with reason codes.

-   **Write-up (README.md)**:

    -   Problem framing; signals; model; pairwise LLM setup; rank aggregation; ablations; limitations; next steps.

    -   Screenshots of top-10 repo cards with one-line investment theses.

-   **Instructions**: `conda env create -f env.yml`, `python -m src.run --token $GITHUB_TOKEN`.