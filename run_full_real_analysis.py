#!/usr/bin/env python3
"""
BSV Repository Prioritizer - Complete Real Analysis
Run the actual algorithm on all 100 repositories from Dataset.csv

This script will:
1. Process the Dataset.csv to extract GitHub URLs
2. Run real GitHub API data collection
3. Perform feature engineering with real metrics
4. Execute LLM pairwise ranking with authentic AI analysis
5. Generate final scores with comprehensive validation
6. Output authentic rankings for all 100 repositories
"""

import sys
import time
import pandas as pd
import numpy as np
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import asyncio
import aiohttp
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import all pipeline components
try:
    from github_collector import GitHubCollector
    from feature_engineer import FeatureEngineer
    from repository_card_generator import RepositoryCardGenerator
    from pair_selector import StrategicPairSelector as PairSelector
    from llm_judge import LLMJudge
    from bradley_terry_ranker import BradleyTerryRanker
    from final_scorer import FinalScorer
    from explainability_analyzer import ExplainabilityAnalyzer
    from evaluation_system import EvaluationSystem
    from bias_detector import BiasDetector
    from output_generator import OutputGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Some components may not be available. Continuing with available functionality...")
    # Set missing components to None for graceful handling
    GitHubCollector = None
    FeatureEngineer = None
    RepositoryCardGenerator = None
    PairSelector = None
    LLMJudge = None
    BradleyTerryRanker = None
    FinalScorer = None
    ExplainabilityAnalyzer = None
    EvaluationSystem = None
    BiasDetector = None
    OutputGenerator = None

class FullDatasetAnalyzer:
    """
    Complete BSV Repository Prioritizer for all 100 repositories
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_logging()
        self.results = {}
        self.start_time = time.time()
        
        print("🚀 BSV REPOSITORY PRIORITIZER - COMPLETE REAL ANALYSIS")
        print("=" * 70)
        print("Running authentic analysis on all 100 repositories")
        print("This will take 15-30 minutes depending on API limits")
        print()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "full_analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FullDatasetAnalyzer')
        
    def extract_github_urls(self, dataset_path: str) -> List[Dict[str, str]]:
        """Extract GitHub repository information from Dataset.csv"""
        self.logger.info("📊 Processing Dataset.csv...")
        
        df = pd.read_csv(dataset_path)
        repositories = []
        
        for idx, row in df.iterrows():
            github_url = row['Name']
            
            # Extract owner/repo from GitHub URL
            match = re.search(r'github\.com/([^/]+)/([^/]+)', github_url.rstrip('/'))
            if not match:
                self.logger.warning(f"Invalid GitHub URL: {github_url}")
                continue
                
            owner, repo = match.groups()
            
            repo_info = {
                'owner': owner,
                'repo': repo,
                'full_name': f"{owner}/{repo}",
                'github_url': github_url,
                'description': row['Description'],
                'dataset_index': idx + 1,
                'initial_stars': int(str(row['Starts']).replace(',', '')) if pd.notna(row['Starts']) else 0,
                'initial_forks': int(str(row['Forks']).replace(',', '')) if pd.notna(row['Forks']) else 0,
                'website': row['Website'] if pd.notna(row['Website']) else ''
            }
            
            repositories.append(repo_info)
            
        self.logger.info(f"✅ Extracted {len(repositories)} valid repositories")
        return repositories
        
    def run_task1_data_collection(self, repositories: List[Dict]) -> pd.DataFrame:
        """Task 1: Real GitHub API data collection"""
        self.logger.info("📋 Task 1: GitHub API Data Collection")
        self.logger.info(f"Collecting data for {len(repositories)} repositories...")
        
        collector = GitHubCollector()
        all_data = []
        
        for i, repo_info in enumerate(repositories, 1):
            self.logger.info(f"Processing {i}/{len(repositories)}: {repo_info['full_name']}")
            
            try:
                # Collect comprehensive repository data
                repo_data = collector.collect_repository_data(
                    repo_info['owner'], 
                    repo_info['repo']
                )
                
                # Add dataset metadata
                repo_data.update({
                    'dataset_index': repo_info['dataset_index'],
                    'dataset_description': repo_info['description'],
                    'dataset_website': repo_info['website']
                })
                
                all_data.append(repo_data)
                
                # Rate limiting
                time.sleep(1.2)  # Respect GitHub API limits
                
            except Exception as e:
                self.logger.error(f"Failed to collect data for {repo_info['full_name']}: {e}")
                # Create minimal entry to avoid losing the repository
                minimal_data = {
                    'repository': repo_info['full_name'],
                    'owner': repo_info['owner'],
                    'repo_name': repo_info['repo'],
                    'stars': repo_info['initial_stars'],
                    'forks': repo_info['initial_forks'],
                    'description': repo_info['description'],
                    'dataset_index': repo_info['dataset_index'],
                    'collection_failed': True
                }
                all_data.append(minimal_data)
                
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save raw data
        output_path = self.project_root / "data" / "full_dataset_raw.csv"
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"✅ Task 1 completed: {len(df)} repositories collected")
        self.logger.info(f"📁 Raw data saved to: {output_path}")
        
        return df
        
    def run_task2_feature_engineering(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Task 2: Real feature engineering with comprehensive metrics"""
        self.logger.info("📋 Task 2: Feature Engineering & Signals")
        
        engineer = FeatureEngineer()
        
        # Process features for all repositories
        processed_df, feature_metadata = engineer.process_features(raw_df)
        
        # Save processed features
        output_path = self.project_root / "data" / "full_dataset_features.csv"
        metadata_path = self.project_root / "data" / "full_dataset_features_metadata.json"
        
        processed_df.to_csv(output_path, index=False)
        
        with open(metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2, default=str)
            
        self.logger.info(f"✅ Task 2 completed: {len(processed_df)} repositories, {len(processed_df.columns)} features")
        self.logger.info(f"📁 Features saved to: {output_path}")
        
        return processed_df
        
    def run_task3_llm_ranking(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Task 3: Real LLM pairwise ranking with authentic AI analysis"""
        self.logger.info("📋 Task 3: LLM Pairwise Ranking System")
        
        # Generate repository cards
        self.logger.info("Generating repository cards for LLM analysis...")
        card_generator = RepositoryCardGenerator()
        repository_cards = {}
        
        for _, row in features_df.iterrows():
            card = card_generator.generate_card(row.to_dict())
            repository_cards[row['repository']] = card
            
        # Save repository cards
        cards_path = self.project_root / "data" / "full_dataset_repository_cards.json"
        with open(cards_path, 'w') as f:
            json.dump(repository_cards, f, indent=2, default=str)
            
        # Select pairs for comparison
        self.logger.info("Selecting repository pairs for LLM comparison...")
        pair_selector = PairSelector()
        
        # For 100 repositories, use strategic sampling
        selected_pairs = pair_selector.select_pairs(
            list(repository_cards.keys()),
            target_comparisons=300,  # Manageable number for 100 repos
            strategy='balanced'
        )
        
        self.logger.info(f"Selected {len(selected_pairs)} pairs for LLM analysis")
        
        # Run LLM judging
        self.logger.info("Running LLM pairwise comparisons...")
        llm_judge = LLMJudge()
        judgments = []
        
        for i, (repo_a, repo_b) in enumerate(selected_pairs, 1):
            if i % 10 == 0:
                self.logger.info(f"LLM comparison progress: {i}/{len(selected_pairs)}")
                
            try:
                judgment = llm_judge.compare_repositories(
                    repository_cards[repo_a],
                    repository_cards[repo_b]
                )
                
                judgments.append({
                    'repo_a': repo_a,
                    'repo_b': repo_b,
                    'winner': judgment['winner'],
                    'reasoning': judgment['reasoning'],
                    'confidence': judgment['confidence']
                })
                
                # Rate limiting for API
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"LLM comparison failed for {repo_a} vs {repo_b}: {e}")
                
        # Save LLM judgments
        judgments_path = self.project_root / "data" / "full_dataset_llm_judgments.json"
        with open(judgments_path, 'w') as f:
            json.dump(judgments, f, indent=2)
            
        # Run Bradley-Terry ranking
        self.logger.info("Computing Bradley-Terry rankings...")
        bt_ranker = BradleyTerryRanker()
        rankings = bt_ranker.compute_rankings(judgments)
        
        # Create rankings DataFrame
        rankings_data = []
        for repo, score in rankings.items():
            rankings_data.append({
                'repository': repo,
                'llm_preference_score': score,
                'ranking_position': len([r for r in rankings.values() if r > score]) + 1
            })
            
        rankings_df = pd.DataFrame(rankings_data)
        
        # Save LLM rankings
        rankings_path = self.project_root / "data" / "full_dataset_llm_rankings.csv"
        rankings_df.to_csv(rankings_path, index=False)
        
        self.logger.info(f"✅ Task 3 completed: {len(rankings_df)} repositories ranked")
        self.logger.info(f"📁 LLM rankings saved to: {rankings_path}")
        
        return rankings_df
        
    def run_task4_final_scoring(self, features_df: pd.DataFrame, rankings_df: pd.DataFrame) -> Dict[str, str]:
        """Task 4: Final scoring with comprehensive evaluation"""
        self.logger.info("📋 Task 4: Final Scoring & Evaluation")
        
        # 4.1 Final Scoring
        self.logger.info("4.1 Computing final composite scores...")
        scorer = FinalScorer()
        
        # Merge features and rankings
        merged_df = features_df.merge(rankings_df, on='repository', how='inner')
        
        # Calculate final scores
        final_scores_df = scorer.calculate_final_scores_from_merged(merged_df)
        
        # Save final scores
        scores_path = self.project_root / "data" / "full_dataset_final_scores.csv"
        final_scores_df.to_csv(scores_path, index=False)
        
        # 4.2 Explainability Analysis
        self.logger.info("4.2 Running explainability analysis...")
        explainer = ExplainabilityAnalyzer()
        explanations = explainer.analyze_explainability(final_scores_df, features_df)
        
        explanations_path = self.project_root / "data" / "full_dataset_explanations.json"
        explainer.save_explanations(explanations, explanations_path)
        
        # 4.3 Comprehensive Evaluation
        self.logger.info("4.3 Running comprehensive evaluation...")
        evaluator = EvaluationSystem()
        
        # Run ablation studies
        ablation_results = evaluator.run_ablation_studies(final_scores_df)
        
        # Run sanity checks
        sanity_results = evaluator.run_sanity_checks(merged_df)
        
        # Run stability analysis
        stability_results = evaluator.run_stability_analysis(final_scores_df)
        
        # Generate evaluation report
        evaluation_path = self.project_root / "data" / "full_dataset_evaluation.json"
        evaluator.generate_evaluation_report(evaluation_path)
        
        # 4.4 Bias Detection
        self.logger.info("4.4 Running bias detection analysis...")
        bias_detector = BiasDetector()
        bias_results = bias_detector.run_comprehensive_bias_analysis(merged_df)
        
        bias_path = self.project_root / "data" / "full_dataset_bias_analysis.json"
        bias_detector.save_bias_analysis(bias_results, bias_path)
        
        self.logger.info("✅ Task 4 completed: Comprehensive scoring and validation")
        
        return {
            'final_scores': scores_path,
            'explanations': explanations_path,
            'evaluation': evaluation_path,
            'bias_analysis': bias_path
        }
        
    def run_task5_output_generation(self, task4_results: Dict[str, str]) -> str:
        """Task 5: Generate investment-grade outputs"""
        self.logger.info("📋 Task 5: Output Generation")
        
        generator = OutputGenerator()
        
        # Load all results
        results = generator.load_all_data(
            scores_path=task4_results['final_scores'],
            explanations_path=task4_results['explanations'],
            evaluation_path=task4_results['evaluation'],
            bias_path=task4_results['bias_analysis'],
            features_path=self.project_root / "data" / "full_dataset_features.csv"
        )
        
        # Generate final CSV
        final_csv_path = self.project_root / "output" / "bsv_complete_dataset_rankings.csv"
        generator.generate_prioritized_csv(results, final_csv_path)
        
        # Generate executive summary
        summary_path = self.project_root / "output" / "bsv_complete_executive_summary.md"
        generator.generate_executive_summary(results, summary_path)
        
        # Generate methodology documentation
        methodology_path = self.project_root / "output" / "bsv_complete_methodology.md"
        generator.generate_methodology_documentation(results, methodology_path)
        
        # Create visualization suite
        viz_dir = self.project_root / "output" / "complete_visualizations"
        generator.create_visualization_suite(results, viz_dir)
        
        # Generate comprehensive PDF report
        pdf_path = self.project_root / "output" / "bsv_complete_analysis_report.pdf"
        generator.generate_pdf_report(results, pdf_path)
        
        self.logger.info("✅ Task 5 completed: All deliverables generated")
        
        return final_csv_path
        
    def run_complete_analysis(self) -> str:
        """Run the complete analysis pipeline on all 100 repositories"""
        try:
            # Extract repository information
            repositories = self.extract_github_urls("Docs/Dataset.csv")
            
            # Task 1: Data Collection
            raw_df = self.run_task1_data_collection(repositories)
            self.results['task1'] = raw_df
            
            # Task 2: Feature Engineering
            features_df = self.run_task2_feature_engineering(raw_df)
            self.results['task2'] = features_df
            
            # Task 3: LLM Ranking
            rankings_df = self.run_task3_llm_ranking(features_df)
            self.results['task3'] = rankings_df
            
            # Task 4: Final Scoring & Evaluation
            task4_results = self.run_task4_final_scoring(features_df, rankings_df)
            self.results['task4'] = task4_results
            
            # Task 5: Output Generation
            final_csv_path = self.run_task5_output_generation(task4_results)
            self.results['task5'] = final_csv_path
            
            # Generate summary
            total_time = time.time() - self.start_time
            
            print()
            print("=" * 70)
            print("📊 COMPLETE REAL ANALYSIS SUMMARY")
            print("=" * 70)
            print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
            print(f"📈 Repositories analyzed: {len(features_df)}")
            print(f"🔬 LLM comparisons: {len(rankings_df)}")
            print(f"📁 Final rankings: {final_csv_path}")
            
            # Show top 10 results
            final_df = pd.read_csv(final_csv_path)
            print()
            print("🏆 Top 10 Repositories (Real Analysis):")
            print("-" * 60)
            
            top_10 = final_df.head(10)
            for i, (_, repo) in enumerate(top_10.iterrows(), 1):
                print(f"{i:2d}. {repo['repository']:<35} | Score: {repo['final_score']:.3f}")
                
            print()
            print("🎉 COMPLETE REAL ANALYSIS FINISHED!")
            print("📊 All 100 repositories analyzed with authentic BSV algorithm")
            
            return final_csv_path
            
        except Exception as e:
            self.logger.error(f"Complete analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main execution function"""
    analyzer = FullDatasetAnalyzer()
    final_results = analyzer.run_complete_analysis()
    return final_results

if __name__ == "__main__":
    main()
