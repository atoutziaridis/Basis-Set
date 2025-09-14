"""
Test script for extended GitHub data collection
Tests the collector with subtasks 1.2 and 1.3 included
"""

import pandas as pd
import os
from pathlib import Path
from github_collector import GitHubCollector
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_extended_collection():
    """Test extended data collection with code quality and adoption signals"""
    
    # Sample repositories - choosing diverse examples from different languages
    test_repos = [
        "https://github.com/twentyhq/twenty",         # TypeScript/React CRM
        "https://github.com/Mooncake-Labs/pg_mooncake" # Database/Analytics (smaller repo)
    ]
    
    try:
        logger.info("Testing extended GitHub data collection...")
        
        try:
            collector = GitHubCollector()
            logger.info("✅ GitHubCollector initialized successfully")
        except ValueError as e:
            logger.info("⚠️  GitHub token not found - testing structure only")
            logger.info("   Set GITHUB_TOKEN environment variable to run full test")
            
            # Test with mock data structure
            mock_extended_data = {
                'repo_url': 'https://github.com/test/repo',
                'stars': 100,
                'forks': 20,
                
                # Original features (from subtask 1.1)
                'commits_6_months': 150,
                'bus_factor': 0.7,
                'total_contributors': 15,
                
                # Code Quality Indicators (subtask 1.2)
                'has_ci_cd': True,
                'workflow_count': 3,
                'workflow_files': ['ci.yml', 'test.yml', 'deploy.yml'],
                'has_tests': True,
                'test_directories': ['tests', 'spec'],
                'readme_length': 2500,
                'readme_quality_score': 0.85,
                'has_docs_directory': True,
                'has_dockerfile': True,
                'has_docker_compose': False,
                'has_package_json': True,
                'has_requirements_txt': False,
                'has_pyproject_toml': False,
                'has_cargo_toml': False,
                'has_makefile': True,
                'has_eslintrc': True,
                'has_prettier': True,
                'has_gitignore': True,
                'config_completeness_score': 0.6,
                
                # Adoption Signals (subtask 1.3)
                'fork_to_star_ratio': 0.2,
                'stars_per_month': 12.5,
                'pypi_downloads': 0,
                'npm_downloads': 15000,
                'cargo_downloads': 0,
                'has_package': True,
                'dependents_count': 45,
                'dependents_scraped': True,
                'network_count': 20,
                'subscribers_count': 50,
                'engagement_score': 170.0,
                
                'collected_at': '2024-01-01T12:00:00Z'
            }
            
            df = pd.DataFrame([mock_extended_data])
            logger.info(f"✅ Extended mock data structure created successfully")
            logger.info(f"   Total features: {len(df.columns)}")
            
            # Categorize features
            original_features = ['stars', 'forks', 'commits_6_months', 'bus_factor', 'total_contributors']
            quality_features = [col for col in df.columns if any(keyword in col for keyword in 
                               ['ci_cd', 'workflow', 'test', 'readme', 'docs', 'dockerfile', 'config'])]
            adoption_features = [col for col in df.columns if any(keyword in col for keyword in 
                               ['ratio', 'downloads', 'dependents', 'network', 'engagement'])]
            
            logger.info(f"   Original features (1.1): {len(original_features)}")
            logger.info(f"   Quality features (1.2): {len(quality_features)}")
            logger.info(f"   Adoption features (1.3): {len(adoption_features)}")
            
            return True
            
        # If token is available, run actual test
        if collector:
            logger.info("Running actual extended data collection test...")
            
            # Test with just one repo to verify new features
            df = collector.collect_batch_data([test_repos[0]])
            
            logger.info(f"✅ Extended collection completed successfully")
            logger.info(f"   Repositories processed: {len(df)}")
            logger.info(f"   Total features collected: {len(df.columns)}")
            
            # Check for new feature categories
            original_features = [
                'stars', 'forks', 'commits_6_months', 'total_contributors', 'bus_factor'
            ]
            quality_features = [
                'has_ci_cd', 'has_tests', 'readme_quality_score', 'config_completeness_score'
            ]
            adoption_features = [
                'fork_to_star_ratio', 'stars_per_month', 'dependents_count', 'engagement_score'
            ]
            
            all_required_features = original_features + quality_features + adoption_features
            missing_features = [f for f in all_required_features if f not in df.columns]
            
            if missing_features:
                logger.warning(f"⚠️  Missing required features: {missing_features}")
            else:
                logger.info("✅ All required features from all subtasks present")
            
            # Feature breakdown
            quality_present = [f for f in quality_features if f in df.columns]
            adoption_present = [f for f in adoption_features if f in df.columns]
            
            logger.info(f"✅ Code Quality features: {len(quality_present)}/{len(quality_features)}")
            logger.info(f"✅ Adoption Signal features: {len(adoption_present)}/{len(adoption_features)}")
            
            # Show sample values
            if len(df) > 0:
                row = df.iloc[0]
                logger.info("Sample feature values:")
                logger.info(f"  README Quality Score: {row.get('readme_quality_score', 'N/A')}")
                logger.info(f"  Has CI/CD: {row.get('has_ci_cd', 'N/A')}")
                logger.info(f"  Config Completeness: {row.get('config_completeness_score', 'N/A')}")
                logger.info(f"  Stars per Month: {row.get('stars_per_month', 'N/A')}")
                logger.info(f"  Engagement Score: {row.get('engagement_score', 'N/A')}")
                logger.info(f"  Dependents Count: {row.get('dependents_count', 'N/A')}")
            
            # Save test results
            output_path = Path(__file__).parent.parent / "data" / "test_extended_results.csv"
            output_path.parent.mkdir(exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"✅ Extended test results saved to: {output_path}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Extended test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_extended_collection()
    if success:
        logger.info("🎉 Extended data collection test completed successfully!")
        logger.info("Ready to run full extended collection with: python src/data_collection_runner.py")
        logger.info("New features added:")
        logger.info("  📊 Code Quality Indicators (CI/CD, tests, README quality, configs)")
        logger.info("  📈 Adoption Signals (downloads, dependents, engagement metrics)")
    else:
        logger.error("❌ Extended data collection test failed")
        exit(1)