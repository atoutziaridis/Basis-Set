"""
Test Task 3 pipeline with a small dataset to verify functionality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.task3_llm_ranking_pipeline import LLMRankingPipeline

def create_test_dataset():
    """Create a small test dataset with multiple repositories"""
    
    # Create synthetic test data with multiple repositories
    test_data = []
    
    repo_configs = [
        {
            'repo_name': 'test-ai-framework', 
            'owner': 'ai-labs',
            'description': 'Revolutionary AI framework for distributed training',
            'stars': 5000, 'forks': 800, 'total_contributors': 50,
            'category_potential_score': 0.85, 'bsv_investment_score': 0.80
        },
        {
            'repo_name': 'blockchain-toolkit', 
            'owner': 'crypto-dev',
            'description': 'Next-generation blockchain development toolkit',
            'stars': 3200, 'forks': 400, 'total_contributors': 25,
            'category_potential_score': 0.75, 'bsv_investment_score': 0.70
        },
        {
            'repo_name': 'database-optimizer', 
            'owner': 'db-team',
            'description': 'High-performance database query optimization engine',
            'stars': 1800, 'forks': 200, 'total_contributors': 15,
            'category_potential_score': 0.65, 'bsv_investment_score': 0.60
        },
        {
            'repo_name': 'web3-platform', 
            'owner': 'web3-builders',
            'description': 'Decentralized application platform with smart contracts',
            'stars': 4500, 'forks': 600, 'total_contributors': 35,
            'category_potential_score': 0.80, 'bsv_investment_score': 0.75
        },
        {
            'repo_name': 'ml-pipeline', 
            'owner': 'data-labs',
            'description': 'Automated machine learning pipeline for production systems',
            'stars': 2100, 'forks': 300, 'total_contributors': 20,
            'category_potential_score': 0.70, 'bsv_investment_score': 0.65
        }
    ]
    
    # Fill in standard fields for all repos
    for config in repo_configs:
        repo_data = {
            # Basic fields
            'repo_url': f"https://github.com/{config['owner']}/{config['repo_name']}",
            'owner': config['owner'],
            'repo_name': config['repo_name'],
            'full_name': f"{config['owner']}/{config['repo_name']}",
            'description': config['description'],
            'stars': config['stars'],
            'forks': config['forks'],
            'watchers': config['stars'],
            'created_at': '2023-01-01T00:00:00+00:00',
            'updated_at': '2024-09-01T00:00:00+00:00',
            'pushed_at': '2024-09-01T00:00:00+00:00',
            'language': 'Python',
            'size': np.random.randint(500, 2000),
            'default_branch': 'main',
            'archived': False,
            'disabled': False,
            'private': False,
            'fork': False,
            'has_issues': True,
            'has_projects': True,
            'has_wiki': False,
            'has_downloads': True,
            'license': 'MIT License',
            'topics': '["ai", "machine-learning"]',
            'open_issues_count': np.random.randint(5, 50),
            'primary_language': 'Python',
            'language_diversity': np.random.randint(2, 6),
            'languages_json': '{"Python": 80.5, "JavaScript": 19.5}',
            
            # Activity metrics
            'commits_6_months': np.random.randint(50, 200),
            'active_weeks_6_months': np.random.randint(15, 25),
            'avg_commits_per_week': np.random.uniform(2, 8),
            'commits_30_days': np.random.randint(5, 30),
            'total_releases': np.random.randint(5, 20),
            'latest_release_date': '2024-08-01T00:00:00+00:00',
            'days_since_last_release': np.random.randint(10, 100),
            'releases_last_year': np.random.randint(3, 12),
            'issues_30_days': np.random.randint(2, 15),
            'prs_30_days': np.random.randint(1, 10),
            'avg_issue_response_time_hours': np.random.uniform(2, 48),
            'median_issue_response_time_hours': np.random.uniform(1, 24),
            
            # Team metrics
            'total_contributors': config['total_contributors'],
            'bus_factor': np.random.uniform(0.3, 0.7),
            'top_contributor_percentage': np.random.uniform(30, 70),
            'contribution_gini': np.random.uniform(0.4, 0.8),
            'active_contributors': np.random.randint(3, 15),
            
            # Quality indicators
            'has_ci_cd': np.random.choice([True, False], p=[0.8, 0.2]),
            'workflow_count': np.random.randint(1, 5),
            'workflow_files': '["ci.yml", "tests.yml"]',
            'has_tests': True,
            'test_directories': '["tests", "test"]',
            'readme_length': np.random.randint(2000, 8000),
            'readme_quality_score': np.random.uniform(0.6, 1.0),
            'has_docs_directory': np.random.choice([True, False]),
            'has_dockerfile': np.random.choice([True, False]),
            'has_docker_compose': np.random.choice([True, False]),
            'has_package_json': np.random.choice([True, False]),
            'has_requirements_txt': True,
            'has_pyproject_toml': np.random.choice([True, False]),
            'has_cargo_toml': False,
            'has_makefile': np.random.choice([True, False]),
            'has_eslintrc': np.random.choice([True, False]),
            'has_prettier': np.random.choice([True, False]),
            'has_gitignore': True,
            'config_completeness_score': np.random.uniform(0.5, 1.0),
            
            # Market metrics
            'fork_to_star_ratio': config['forks'] / max(config['stars'], 1),
            'stars_per_month': np.random.uniform(50, 300),
            'pypi_downloads': np.random.randint(1000, 50000),
            'npm_downloads': 0,
            'cargo_downloads': 0,
            'has_package': True,
            'dependents_count': np.random.randint(10, 100),
            'dependents_scraped': True,
            'network_count': config['forks'],
            'subscribers_count': np.random.randint(20, 200),
            'engagement_score': np.random.uniform(0.4, 0.9),
            
            # Funding detection (all unfunded for simplicity)
            'total_direct_funding_indicators': 0,
            'total_investor_mentions_indicators': 0,
            'total_financial_terms_indicators': 0,
            'total_company_stage_indicators': 0,
            'total_funding_indicators': 0,
            'funding_indicators_by_source': '{}',
            'negative_funding_indicators': 0,
            'strong_positive_indicators': 0,
            'funding_confidence': 0.0,
            'funding_risk_level': 'low_risk_unfunded',
            'text_sources_analyzed': 4,
            'collected_at': '2024-09-14T12:00:00',
            
            # Engineered scores (from config)
            'commit_velocity_score': np.random.uniform(0.4, 0.9),
            'release_cadence_score': np.random.uniform(0.4, 0.9),
            'maintenance_activity_score': np.random.uniform(0.4, 0.9),
            'development_consistency_score': np.random.uniform(0.4, 0.9),
            'execution_velocity_composite': np.random.uniform(0.4, 0.9),
            'team_resilience_score': np.random.uniform(0.4, 0.9),
            'community_health_score': np.random.uniform(0.4, 0.9),
            'growth_trajectory_score': np.random.uniform(0.4, 0.9),
            'network_effects_score': np.random.uniform(0.4, 0.9),
            'team_community_composite': np.random.uniform(0.4, 0.9),
            'operational_readiness_score': np.random.uniform(0.4, 0.9),
            'code_quality_score': np.random.uniform(0.4, 0.9),
            'api_stability_score': np.random.uniform(0.4, 0.9),
            'documentation_score': np.random.uniform(0.4, 0.9),
            'technical_maturity_composite': np.random.uniform(0.4, 0.9),
            'problem_ambition_score': np.random.uniform(0.4, 0.9),
            'commercial_viability_score': np.random.uniform(0.4, 0.9),
            'technology_differentiation_score': np.random.uniform(0.4, 0.9),
            'market_readiness_score': np.random.uniform(0.4, 0.9),
            'market_positioning_composite': np.random.uniform(0.4, 0.9),
            'category_potential_score': config['category_potential_score'],
            'bsv_investment_score': config['bsv_investment_score']
        }
        
        test_data.append(repo_data)
    
    return pd.DataFrame(test_data)

def run_test():
    """Run Task 3 pipeline test with small dataset"""
    
    print("🧪 Creating test dataset...")
    test_df = create_test_dataset()
    
    # Save test dataset
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    test_path = data_dir / "test_task3_dataset.csv"
    test_df.to_csv(str(test_path), index=False)
    
    print(f"✅ Created test dataset with {len(test_df)} repositories")
    print(f"📁 Saved to: {test_path}")
    
    # Run pipeline
    print("\n🚀 Running Task 3 pipeline test...")
    
    pipeline = LLMRankingPipeline(
        target_comparisons=10,  # Small number for testing
        openai_model="gpt-3.5-turbo"
    )
    
    try:
        final_rankings = pipeline.run_complete_pipeline(str(test_path))
        
        print(f"\n🎉 Test completed successfully!")
        print(f"📊 Final rankings:")
        print(final_rankings[['final_rank', 'repository', 'integrated_score', 'bradley_terry_score']])
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    if success:
        print("\n✅ Task 3 pipeline test passed!")
    else:
        print("\n❌ Task 3 pipeline test failed!")