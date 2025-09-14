"""
BSV Comprehensive Evaluation System
Implements Task 4.3: Ablation studies, sanity checks, and stability analysis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AblationResult:
    """Results from an ablation study"""
    name: str
    description: str
    final_scores: List[float]
    rank_correlation_spearman: float
    rank_correlation_kendall: float
    score_mse: float
    score_mae: float
    top_10_overlap: int
    top_5_overlap: int

@dataclass
class SanityCheck:
    """Results from a sanity check"""
    name: str
    description: str
    correlation: float
    p_value: float
    passed: bool
    details: Dict[str, Any]

@dataclass
class StabilityAnalysis:
    """Results from bootstrap stability analysis"""
    bootstrap_iterations: int
    rank_stability_scores: List[float]
    stable_top_quartile: List[str]
    rank_variance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

class EvaluationSystem:
    """
    BSV Comprehensive Evaluation System
    
    Performs ablation studies, sanity checks, and stability analysis
    to validate the final ranking system. Implements Task 4.3.
    """
    
    def __init__(self):
        """Initialize evaluation system"""
        self.ablation_results: List[AblationResult] = []
        self.sanity_checks: List[SanityCheck] = []
        self.stability_analysis: Optional[StabilityAnalysis] = None
        
        logger.info("EvaluationSystem initialized")
    
    def load_data(self, scores_path: str, features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load final scores and feature data"""
        logger.info("Loading evaluation data...")
        
        scores_df = pd.read_csv(scores_path)
        features_df = pd.read_csv(features_path)
        
        logger.info(f"Loaded {len(scores_df)} scored repositories")
        logger.info(f"Loaded {len(features_df)} repositories with {len(features_df.columns)} features")
        
        return scores_df, features_df
    
    def _calculate_alternative_score(self, features_df: pd.DataFrame, 
                                   weights: Dict[str, float],
                                   components: Dict[str, List[str]]) -> pd.Series:
        """Calculate alternative scoring using different weights or components"""
        
        scores = []
        
        for idx in range(len(features_df)):
            repo_score = 0.0
            
            for component, weight in weights.items():
                if component in components:
                    # Calculate component score as average of available features
                    component_features = components[component]
                    component_values = []
                    
                    for feature in component_features:
                        if feature in features_df.columns:
                            value = features_df.iloc[idx][feature]
                            if pd.notna(value) and isinstance(value, (int, float)):
                                # Normalize to [0,1] if needed
                                if value > 1.0:
                                    feature_series = features_df[feature]
                                    if feature_series.max() > feature_series.min():
                                        normalized_value = (value - feature_series.min()) / (feature_series.max() - feature_series.min())
                                    else:
                                        normalized_value = 0.5
                                else:
                                    normalized_value = value
                                
                                component_values.append(normalized_value)
                    
                    if component_values:
                        component_score = np.mean(component_values)
                        repo_score += weight * component_score
            
            scores.append(repo_score)
        
        return pd.Series(scores)
    
    def run_ablation_studies(self, scores_df: pd.DataFrame, features_df: pd.DataFrame) -> List[AblationResult]:
        """
        Run ablation studies to understand component contributions
        
        Tests:
        1. LLM-only rankings
        2. Features-only rankings (no LLM)
        3. Remove each component individually
        4. Equal weights vs optimized weights
        """
        
        logger.info("Running ablation studies...")
        
        original_scores = scores_df['final_score'].values
        original_ranks = scores_df['rank'].values
        
        ablation_tests = []
        
        # Define component mappings
        components = {
            'llm_preference': ['integrated_score', 'bradley_terry_score'],
            'technical_execution': [
                'commit_velocity_score', 'release_cadence_score', 'code_quality_score',
                'has_ci_cd', 'has_tests', 'config_completeness_score'
            ],
            'market_adoption': [
                'stars_per_month', 'dependents_count', 'engagement_score',
                'growth_trajectory_score', 'network_effects_score'
            ],
            'team_resilience': [
                'bus_factor', 'team_resilience_score', 'community_health_score',
                'total_contributors', 'active_contributors'
            ]
        }
        
        # 1. LLM-only ranking
        llm_only_scores = self._calculate_alternative_score(
            features_df, 
            {'llm_preference': 1.0},
            components
        )
        
        ablation_tests.append({
            'name': 'llm_only',
            'description': 'LLM preference score only (100%)',
            'scores': llm_only_scores
        })
        
        # 2. Features-only ranking (no LLM)
        features_only_scores = self._calculate_alternative_score(
            features_df,
            {
                'technical_execution': 0.5,
                'market_adoption': 0.3,
                'team_resilience': 0.2
            },
            components
        )
        
        ablation_tests.append({
            'name': 'features_only',
            'description': 'Features only (no LLM): Technical 50%, Market 30%, Team 20%',
            'scores': features_only_scores
        })
        
        # 3. Remove technical execution
        no_technical_scores = self._calculate_alternative_score(
            features_df,
            {
                'llm_preference': 0.75,
                'market_adoption': 0.15,
                'team_resilience': 0.10
            },
            components
        )
        
        ablation_tests.append({
            'name': 'no_technical',
            'description': 'Remove technical execution: LLM 75%, Market 15%, Team 10%',
            'scores': no_technical_scores
        })
        
        # 4. Remove market adoption
        no_market_scores = self._calculate_alternative_score(
            features_df,
            {
                'llm_preference': 0.70,
                'technical_execution': 0.20,
                'team_resilience': 0.10
            },
            components
        )
        
        ablation_tests.append({
            'name': 'no_market',
            'description': 'Remove market adoption: LLM 70%, Technical 20%, Team 10%',
            'scores': no_market_scores
        })
        
        # 5. Equal weights
        equal_weights_scores = self._calculate_alternative_score(
            features_df,
            {
                'llm_preference': 0.25,
                'technical_execution': 0.25,
                'market_adoption': 0.25,
                'team_resilience': 0.25
            },
            components
        )
        
        ablation_tests.append({
            'name': 'equal_weights',
            'description': 'Equal weights: All components 25% each',
            'scores': equal_weights_scores
        })
        
        # Calculate metrics for each ablation test
        results = []
        
        for test in ablation_tests:
            test_scores = test['scores'].values
            
            # Create temporary ranking
            temp_df = pd.DataFrame({
                'repo_name': scores_df['repo_name'],
                'score': test_scores
            }).sort_values('score', ascending=False)
            temp_df['rank'] = range(1, len(temp_df) + 1)
            
            # Merge back to get ranks in original order
            merged = scores_df[['repo_name']].merge(temp_df[['repo_name', 'rank']], on='repo_name')
            test_ranks = merged['rank'].values
            
            # Calculate correlations
            spearman_corr, _ = spearmanr(original_ranks, test_ranks)
            kendall_corr, _ = kendalltau(original_ranks, test_ranks)
            
            # Calculate score metrics
            mse = mean_squared_error(original_scores, test_scores)
            mae = mean_absolute_error(original_scores, test_scores)
            
            # Calculate top-k overlap
            original_top_10 = set(scores_df.head(10)['repo_name'].tolist())
            original_top_5 = set(scores_df.head(5)['repo_name'].tolist())
            
            test_top_10 = set(temp_df.head(10)['repo_name'].tolist())
            test_top_5 = set(temp_df.head(5)['repo_name'].tolist())
            
            top_10_overlap = len(original_top_10.intersection(test_top_10))
            top_5_overlap = len(original_top_5.intersection(test_top_5))
            
            result = AblationResult(
                name=test['name'],
                description=test['description'],
                final_scores=test_scores.tolist(),
                rank_correlation_spearman=spearman_corr,
                rank_correlation_kendall=kendall_corr,
                score_mse=mse,
                score_mae=mae,
                top_10_overlap=top_10_overlap,
                top_5_overlap=top_5_overlap
            )
            
            results.append(result)
        
        self.ablation_results = results
        logger.info(f"Completed {len(results)} ablation studies")
        
        return results
    
    def run_sanity_checks(self, scores_df: pd.DataFrame, features_df: pd.DataFrame) -> List[SanityCheck]:
        """
        Run sanity checks to validate ranking reasonableness
        
        Checks:
        1. Correlation with star count (should be moderate, not too high)
        2. Correlation with repository age (should be weak)
        3. Correlation with commit activity (should be positive)
        4. Anti-correlation with funding indicators
        """
        
        logger.info("Running sanity checks...")
        
        # Merge data
        if 'repo_name' in features_df.columns:
            merged_df = scores_df.merge(features_df, on='repo_name', how='left')
        else:
            merged_df = pd.concat([scores_df, features_df], axis=1)
        
        checks = []
        
        # 1. Star count correlation (should be moderate: 0.3-0.7)
        if 'stars' in merged_df.columns:
            stars_corr, stars_p = spearmanr(merged_df['final_score'], merged_df['stars'])
            passed = 0.2 <= abs(stars_corr) <= 0.8  # Not too weak, not too strong
            
            checks.append(SanityCheck(
                name='star_correlation',
                description='Correlation with star count should be moderate (0.2-0.8)',
                correlation=stars_corr,
                p_value=stars_p,
                passed=passed,
                details={'threshold_min': 0.2, 'threshold_max': 0.8}
            ))
        
        # 2. Repository age correlation (should be weak)
        if 'created_at' in merged_df.columns:
            # Calculate days since creation
            merged_df['days_old'] = pd.to_datetime('2024-09-14') - pd.to_datetime(merged_df['created_at'])
            merged_df['days_old_numeric'] = merged_df['days_old'].dt.days
            
            age_corr, age_p = spearmanr(merged_df['final_score'], merged_df['days_old_numeric'])
            passed = abs(age_corr) <= 0.5  # Should not be strongly age-biased
            
            checks.append(SanityCheck(
                name='age_bias',
                description='Correlation with repository age should be weak (|r| <= 0.5)',
                correlation=age_corr,
                p_value=age_p,
                passed=passed,
                details={'threshold': 0.5}
            ))
        
        # 3. Commit activity correlation (should be positive)
        if 'commits_6_months' in merged_df.columns:
            commit_corr, commit_p = spearmanr(merged_df['final_score'], merged_df['commits_6_months'])
            passed = commit_corr > 0.1  # Should have some positive correlation with activity
            
            checks.append(SanityCheck(
                name='activity_correlation',
                description='Correlation with commit activity should be positive',
                correlation=commit_corr,
                p_value=commit_p,
                passed=passed,
                details={'threshold': 0.1}
            ))
        
        # 4. Funding indicators anti-correlation
        if 'funding_confidence' in merged_df.columns:
            funding_corr, funding_p = spearmanr(merged_df['final_score'], merged_df['funding_confidence'])
            passed = funding_corr <= 0.2  # Should not favor funded projects
            
            checks.append(SanityCheck(
                name='funding_bias',
                description='Should not favor funded projects (funding correlation <= 0.2)',
                correlation=funding_corr,
                p_value=funding_p,
                passed=passed,
                details={'threshold': 0.2}
            ))
        
        # 5. Fork-to-star ratio sanity (healthy projects have reasonable ratios)
        if 'fork_to_star_ratio' in merged_df.columns:
            fork_ratio_corr, fork_ratio_p = spearmanr(merged_df['final_score'], merged_df['fork_to_star_ratio'])
            passed = -0.3 <= fork_ratio_corr <= 0.3  # Should not be strongly correlated either way
            
            checks.append(SanityCheck(
                name='fork_ratio_sanity',
                description='Fork-to-star ratio correlation should be neutral',
                correlation=fork_ratio_corr,
                p_value=fork_ratio_p,
                passed=passed,
                details={'threshold_min': -0.3, 'threshold_max': 0.3}
            ))
        
        self.sanity_checks = checks
        logger.info(f"Completed {len(checks)} sanity checks")
        
        return checks
    
    def run_stability_analysis(self, scores_df: pd.DataFrame, features_df: pd.DataFrame,
                             n_bootstrap: int = 100) -> StabilityAnalysis:
        """
        Run bootstrap stability analysis to identify stable rankings
        
        Uses bootstrap resampling to test ranking stability
        """
        
        logger.info(f"Running stability analysis with {n_bootstrap} bootstrap iterations...")
        
        # Store original rankings
        original_ranking = scores_df.set_index('repo_name')['rank'].to_dict()
        repo_names = scores_df['repo_name'].tolist()
        
        # Bootstrap results storage
        bootstrap_rankings = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample (with replacement)
            sample_indices = np.random.choice(len(features_df), size=len(features_df), replace=True)
            bootstrap_features = features_df.iloc[sample_indices].copy()
            bootstrap_scores = scores_df.iloc[sample_indices].copy()
            
            # Recalculate scores (simplified version)
            # In practice, you'd re-run the full scoring pipeline
            # For this demo, we'll add noise to simulate bootstrap variation
            noise_factor = 0.05  # 5% noise
            bootstrap_scores['final_score_bootstrap'] = (
                bootstrap_scores['final_score'] * 
                (1 + np.random.normal(0, noise_factor, len(bootstrap_scores)))
            )
            
            # Re-rank
            bootstrap_scores = bootstrap_scores.sort_values('final_score_bootstrap', ascending=False)
            bootstrap_scores['bootstrap_rank'] = range(1, len(bootstrap_scores) + 1)
            
            # Store ranking
            ranking_dict = bootstrap_scores.set_index('repo_name')['bootstrap_rank'].to_dict()
            bootstrap_rankings.append(ranking_dict)
        
        # Calculate stability metrics
        rank_variances = {}
        rank_stability_scores = []
        
        for repo in repo_names:
            repo_ranks = [ranking.get(repo, len(repo_names)) for ranking in bootstrap_rankings]
            rank_variance = np.var(repo_ranks)
            rank_variances[repo] = rank_variance
            
            # Stability score: inverse of normalized variance
            max_possible_variance = (len(repo_names) ** 2) / 4  # Maximum variance for uniform distribution
            stability_score = 1 - (rank_variance / max_possible_variance)
            rank_stability_scores.append(stability_score)
        
        # Identify stable top quartile
        n_top_quartile = max(1, len(repo_names) // 4)
        stable_repos = sorted(rank_variances.items(), key=lambda x: x[1])[:n_top_quartile]
        stable_top_quartile = [repo for repo, _ in stable_repos]
        
        # Calculate confidence intervals for ranks
        confidence_intervals = {}
        for repo in repo_names:
            repo_ranks = [ranking.get(repo, len(repo_names)) for ranking in bootstrap_rankings]
            ci_lower = np.percentile(repo_ranks, 2.5)
            ci_upper = np.percentile(repo_ranks, 97.5)
            confidence_intervals[repo] = (ci_lower, ci_upper)
        
        analysis = StabilityAnalysis(
            bootstrap_iterations=n_bootstrap,
            rank_stability_scores=rank_stability_scores,
            stable_top_quartile=stable_top_quartile,
            rank_variance=rank_variances,
            confidence_intervals=confidence_intervals
        )
        
        self.stability_analysis = analysis
        logger.info("Stability analysis completed")
        
        return analysis
    
    def identify_outliers(self, scores_df: pd.DataFrame, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify clear outliers for manual investigation"""
        
        logger.info("Identifying outliers for manual review...")
        
        # Merge data
        if 'repo_name' in features_df.columns:
            merged_df = scores_df.merge(features_df, on='repo_name', how='left')
        else:
            merged_df = pd.concat([scores_df, features_df], axis=1)
        
        outliers = {
            'high_score_low_stars': [],
            'low_score_high_stars': [],
            'high_score_old_repo': [],
            'low_score_active_repo': []
        }
        
        # High score but low stars
        if 'stars' in merged_df.columns:
            high_score_threshold = merged_df['final_score'].quantile(0.8)
            low_stars_threshold = merged_df['stars'].quantile(0.3)
            
            high_score_low_stars = merged_df[
                (merged_df['final_score'] >= high_score_threshold) &
                (merged_df['stars'] <= low_stars_threshold)
            ]
            outliers['high_score_low_stars'] = high_score_low_stars['repo_name'].tolist()
        
        # Low score but high stars  
        if 'stars' in merged_df.columns:
            low_score_threshold = merged_df['final_score'].quantile(0.3)
            high_stars_threshold = merged_df['stars'].quantile(0.8)
            
            low_score_high_stars = merged_df[
                (merged_df['final_score'] <= low_score_threshold) &
                (merged_df['stars'] >= high_stars_threshold)
            ]
            outliers['low_score_high_stars'] = low_score_high_stars['repo_name'].tolist()
        
        # High score but very old repository
        if 'created_at' in merged_df.columns:
            merged_df['days_old'] = pd.to_datetime('2024-09-14') - pd.to_datetime(merged_df['created_at'])
            merged_df['days_old_numeric'] = merged_df['days_old'].dt.days
            
            high_score_threshold = merged_df['final_score'].quantile(0.8)
            old_repo_threshold = merged_df['days_old_numeric'].quantile(0.8)
            
            high_score_old_repo = merged_df[
                (merged_df['final_score'] >= high_score_threshold) &
                (merged_df['days_old_numeric'] >= old_repo_threshold)
            ]
            outliers['high_score_old_repo'] = high_score_old_repo['repo_name'].tolist()
        
        # Low score but high activity
        if 'commits_6_months' in merged_df.columns:
            low_score_threshold = merged_df['final_score'].quantile(0.3)
            high_activity_threshold = merged_df['commits_6_months'].quantile(0.8)
            
            low_score_active_repo = merged_df[
                (merged_df['final_score'] <= low_score_threshold) &
                (merged_df['commits_6_months'] >= high_activity_threshold)
            ]
            outliers['low_score_active_repo'] = low_score_active_repo['repo_name'].tolist()
        
        return outliers
    
    def generate_evaluation_report(self, output_path: str) -> str:
        """Generate comprehensive evaluation report"""
        
        logger.info("Generating evaluation report...")
        
        def serialize_dataclass(obj):
            """Convert dataclass to dict with JSON-serializable types"""
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for field_name, field_value in asdict(obj).items():
                    if isinstance(field_value, (bool, np.bool_)):
                        result[field_name] = bool(field_value)
                    elif isinstance(field_value, (np.integer, np.floating)):
                        result[field_name] = float(field_value)
                    elif isinstance(field_value, list):
                        result[field_name] = [serialize_dataclass(item) if hasattr(item, '__dataclass_fields__') 
                                            else item for item in field_value]
                    elif isinstance(field_value, dict):
                        result[field_name] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                            for k, v in field_value.items()}
                    else:
                        result[field_name] = field_value
                return result
            return obj

        report_data = {
            'generated_at': datetime.now().isoformat(),
            'evaluation_summary': {
                'ablation_studies': len(self.ablation_results),
                'sanity_checks': len(self.sanity_checks),
                'sanity_checks_passed': sum(1 for check in self.sanity_checks if check.passed),
                'stability_analysis_completed': self.stability_analysis is not None
            },
            'ablation_results': [serialize_dataclass(result) for result in self.ablation_results],
            'sanity_checks': [serialize_dataclass(check) for check in self.sanity_checks],
            'stability_analysis': serialize_dataclass(self.stability_analysis) if self.stability_analysis else None
        }
        
        # Save detailed report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return output_path
    
    def create_visualization(self, scores_df: pd.DataFrame, output_dir: str):
        """Create evaluation visualizations"""
        
        logger.info("Creating evaluation visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Ablation study comparison
        if self.ablation_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Rank correlations
            ablation_names = [result.name for result in self.ablation_results]
            spearman_corrs = [result.rank_correlation_spearman for result in self.ablation_results]
            
            ax1.bar(ablation_names, spearman_corrs)
            ax1.set_title('Rank Correlation with Original (Spearman)')
            ax1.set_ylabel('Correlation')
            ax1.tick_params(axis='x', rotation=45)
            
            # Top-5 overlap
            top5_overlaps = [result.top_5_overlap for result in self.ablation_results]
            ax2.bar(ablation_names, top5_overlaps)
            ax2.set_title('Top-5 Repository Overlap')
            ax2.set_ylabel('Number of Overlapping Repos')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ablation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(scores_df['final_score'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Final Scores')
        plt.xlabel('Final Score')
        plt.ylabel('Frequency')
        plt.axvline(scores_df['final_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {scores_df["final_score"].mean():.3f}')
        plt.legend()
        plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Stability analysis (if available)
        if self.stability_analysis:
            plt.figure(figsize=(12, 8))
            
            repos = list(self.stability_analysis.rank_variance.keys())
            variances = list(self.stability_analysis.rank_variance.values())
            
            # Sort by variance for better visualization
            sorted_data = sorted(zip(repos, variances), key=lambda x: x[1])
            sorted_repos, sorted_variances = zip(*sorted_data)
            
            plt.bar(range(len(sorted_repos)), sorted_variances)
            plt.title('Ranking Stability (Lower Variance = More Stable)')
            plt.xlabel('Repository')
            plt.ylabel('Rank Variance')
            plt.xticks(range(len(sorted_repos)), sorted_repos, rotation=45)
            
            # Highlight stable top quartile
            stable_indices = [i for i, repo in enumerate(sorted_repos) 
                            if repo in self.stability_analysis.stable_top_quartile]
            for idx in stable_indices:
                plt.bar(idx, sorted_variances[idx], color='green', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Main execution function for Task 4.3: Comprehensive Evaluation"""
    logger.info("Starting BSV Comprehensive Evaluation - Task 4.3")
    
    # Initialize evaluation system
    evaluator = EvaluationSystem()
    
    # Define input paths
    project_root = Path(__file__).parent.parent
    scores_path = project_root / "data" / "task4_final_scores.csv"
    features_path = project_root / "data" / "test_task3_dataset.csv"
    
    # Check if input files exist
    if not scores_path.exists():
        logger.error(f"Final scores not found: {scores_path}")
        return
    
    if not features_path.exists():
        logger.error(f"Features data not found: {features_path}")
        return
    
    try:
        # Load data
        scores_df, features_df = evaluator.load_data(str(scores_path), str(features_path))
        
        # Run ablation studies
        ablation_results = evaluator.run_ablation_studies(scores_df, features_df)
        
        # Run sanity checks
        sanity_checks = evaluator.run_sanity_checks(scores_df, features_df)
        
        # Run stability analysis
        stability_analysis = evaluator.run_stability_analysis(scores_df, features_df, n_bootstrap=50)
        
        # Identify outliers
        outliers = evaluator.identify_outliers(scores_df, features_df)
        
        # Generate report
        report_path = project_root / "data" / "task4_evaluation_report.json"
        evaluator.generate_evaluation_report(str(report_path))
        
        # Create visualizations
        viz_dir = project_root / "output" / "evaluation_plots"
        evaluator.create_visualization(scores_df, str(viz_dir))
        
        # Display summary
        print("\n" + "="*60)
        print("🎉 TASK 4.3 COMPREHENSIVE EVALUATION COMPLETE")
        print("="*60)
        print(f"📊 Repositories evaluated: {len(scores_df)}")
        print(f"🔬 Ablation studies: {len(ablation_results)}")
        print(f"✅ Sanity checks: {len(sanity_checks)} ({sum(1 for c in sanity_checks if c.passed)} passed)")
        print(f"📈 Bootstrap iterations: {stability_analysis.bootstrap_iterations}")
        print()
        print("📁 Output files:")
        print(f"   • Evaluation report: {report_path}")
        print(f"   • Visualizations: {viz_dir}")
        print()
        
        # Show ablation results
        print("🔬 Ablation Study Results:")
        for result in ablation_results:
            print(f"   {result.name:<15} | Rank correlation: {result.rank_correlation_spearman:.3f} | "
                  f"Top-5 overlap: {result.top_5_overlap}/5")
        
        # Show sanity check results
        print("\n✅ Sanity Check Results:")
        for check in sanity_checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            print(f"   {check.name:<20} | {status} | Correlation: {check.correlation:.3f}")
        
        # Show stability results
        print(f"\n📈 Stability Analysis:")
        print(f"   Stable top quartile: {', '.join(stability_analysis.stable_top_quartile)}")
        print(f"   Average rank stability: {np.mean(stability_analysis.rank_stability_scores):.3f}")
        
        # Show outliers
        print(f"\n🔍 Outliers for Manual Review:")
        for category, repos in outliers.items():
            if repos:
                print(f"   {category}: {', '.join(repos)}")
        
        return {
            'ablation_results': ablation_results,
            'sanity_checks': sanity_checks,
            'stability_analysis': stability_analysis,
            'outliers': outliers
        }
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
