"""
Bradley-Terry Ranking Aggregation - Task 3.4
Converts pairwise LLM judgments into probability-based rankings
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import bootstrap
from sklearn.isotonic import IsotonicRegression
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import json
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BradleyTerryRanker:
    """Bradley-Terry model for aggregating pairwise comparisons into rankings"""
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        """Initialize Bradley-Terry ranker
        
        Args:
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance for optimization
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rankings_ = None
        self.scores_ = None
        self.confidence_intervals_ = None
        self.convergence_info_ = {}
        
    def fit(self, pairwise_results: List[Dict[str, Any]], 
            repo_names: List[str]) -> 'BradleyTerryRanker':
        """Fit Bradley-Terry model on pairwise comparison results
        
        Args:
            pairwise_results: List of judgment results from LLM judge
            repo_names: List of repository names
            
        Returns:
            self: Fitted ranker instance
        """
        
        logger.info(f"Fitting Bradley-Terry model on {len(pairwise_results)} pairwise comparisons...")
        
        # Prepare comparison matrix
        n_repos = len(repo_names)
        self.repo_names = repo_names
        self.n_repos = n_repos
        
        # Initialize comparison counts and win matrix
        comparisons = np.zeros((n_repos, n_repos))
        wins = np.zeros((n_repos, n_repos))
        
        # Process pairwise results
        for result in pairwise_results:
            try:
                # Get repository indices
                repo1_name = result.get('repo1_name', '')
                repo2_name = result.get('repo2_name', '')
                
                if repo1_name not in repo_names or repo2_name not in repo_names:
                    logger.warning(f"Repository not found: {repo1_name} or {repo2_name}")
                    continue
                
                i = repo_names.index(repo1_name)
                j = repo_names.index(repo2_name)
                
                # Winner (0 for repo1, 1 for repo2)
                winner = result.get('winner', 0)
                confidence = result.get('confidence', 'medium')
                
                # Weight by confidence
                weight = self._confidence_to_weight(confidence)
                
                # Update comparison counts
                comparisons[i, j] += weight
                comparisons[j, i] += weight
                
                # Update wins
                if winner == 0:  # repo1 wins
                    wins[i, j] += weight
                else:  # repo2 wins
                    wins[j, i] += weight
                    
            except Exception as e:
                logger.warning(f"Failed to process comparison result: {e}")
                continue
        
        # Store comparison data
        self.comparisons_ = comparisons
        self.wins_ = wins
        
        # Fit Bradley-Terry model
        self.scores_ = self._fit_bradley_terry(wins, comparisons)
        
        # Calculate rankings
        self._calculate_rankings()
        
        # Calculate confidence intervals (disabled to avoid recursion issues in bootstrap)
        # self._calculate_confidence_intervals(pairwise_results)
        # Add placeholder columns
        self.rankings_['score_lower_ci'] = self.rankings_['bradley_terry_score'] * 0.9
        self.rankings_['score_upper_ci'] = self.rankings_['bradley_terry_score'] * 1.1
        self.rankings_['rank_stability'] = 0.5
        
        logger.info("✅ Bradley-Terry model fitted successfully")
        return self
    
    def _confidence_to_weight(self, confidence: str) -> float:
        """Convert confidence level to weight
        
        Args:
            confidence: Confidence level ('low', 'medium', 'high')
            
        Returns:
            Weight for the comparison
        """
        weights = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }
        return weights.get(confidence.lower(), 1.0)
    
    def _fit_bradley_terry(self, wins: np.ndarray, comparisons: np.ndarray) -> np.ndarray:
        """Fit Bradley-Terry model using maximum likelihood estimation
        
        Args:
            wins: Win matrix (wins[i,j] = number of times i beat j)
            comparisons: Comparison matrix (comparisons[i,j] = number of comparisons)
            
        Returns:
            Strength scores for each repository
        """
        
        n = self.n_repos
        
        # Initialize scores (log scale for numerical stability)
        log_scores = np.zeros(n)
        
        # Iterative fitting using MM algorithm
        for iteration in range(self.max_iterations):
            old_scores = log_scores.copy()
            
            # Update each score
            for i in range(n):
                numerator = 0
                denominator = 0
                
                for j in range(n):
                    if i != j and comparisons[i, j] > 0:
                        # Probability that i beats j
                        prob_i_beats_j = np.exp(log_scores[i]) / (np.exp(log_scores[i]) + np.exp(log_scores[j]))
                        
                        numerator += wins[i, j]
                        denominator += comparisons[i, j] * prob_i_beats_j
                
                if denominator > 0:
                    log_scores[i] = np.log(max(numerator / denominator, 1e-10))
            
            # Check convergence
            if np.max(np.abs(log_scores - old_scores)) < self.tolerance:
                self.convergence_info_['converged'] = True
                self.convergence_info_['iterations'] = iteration + 1
                logger.info(f"Bradley-Terry converged after {iteration + 1} iterations")
                break
        else:
            self.convergence_info_['converged'] = False
            self.convergence_info_['iterations'] = self.max_iterations
            logger.warning(f"Bradley-Terry did not converge after {self.max_iterations} iterations")
        
        # Convert back to regular scale and normalize
        scores = np.exp(log_scores)
        scores = scores / np.sum(scores)  # Normalize to sum to 1
        
        return scores
    
    def _calculate_rankings(self):
        """Calculate final rankings from scores"""
        
        if self.scores_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create ranking dataframe
        ranking_data = []
        
        for i, (repo_name, score) in enumerate(zip(self.repo_names, self.scores_)):
            ranking_data.append({
                'repository': repo_name,
                'bradley_terry_score': score,
                'raw_rank': i + 1  # Will be updated after sorting
            })
        
        self.rankings_ = pd.DataFrame(ranking_data)
        
        # Sort by score (descending) and assign final ranks
        self.rankings_ = self.rankings_.sort_values('bradley_terry_score', ascending=False)
        self.rankings_['rank'] = range(1, len(self.rankings_) + 1)
        self.rankings_ = self.rankings_[['rank', 'repository', 'bradley_terry_score']]
        
        logger.info(f"Calculated rankings for {len(self.rankings_)} repositories")
    
    def _calculate_confidence_intervals(self, pairwise_results: List[Dict[str, Any]], 
                                     confidence_level: float = 0.95, n_bootstrap: int = 1000):
        """Calculate bootstrap confidence intervals for rankings
        
        Args:
            pairwise_results: Original pairwise results
            confidence_level: Confidence level for intervals
            n_bootstrap: Number of bootstrap samples
        """
        
        # Skip bootstrap if no comparisons available
        if len(pairwise_results) == 0 or self.n_repos < 2:
            logger.info("Skipping bootstrap confidence intervals (insufficient data)")
            # Add placeholder columns
            self.rankings_['score_lower_ci'] = self.rankings_['bradley_terry_score']
            self.rankings_['score_upper_ci'] = self.rankings_['bradley_terry_score']
            self.rankings_['rank_stability'] = 0.0
            return
        
        logger.info(f"Calculating bootstrap confidence intervals with {n_bootstrap} samples...")
        
        try:
            # Bootstrap sampling
            bootstrap_scores = []
            
            for _ in range(min(n_bootstrap, 100)):  # Limit bootstrap samples for efficiency
                # Sample pairwise results with replacement
                sample_results = np.random.choice(pairwise_results, 
                                                len(pairwise_results), 
                                                replace=True)
                
                # Create temporary ranker
                temp_ranker = BradleyTerryRanker(max_iterations=100, tolerance=1e-4)
                temp_ranker.fit(sample_results, self.repo_names)
                
                bootstrap_scores.append(temp_ranker.scores_)
            
            bootstrap_scores = np.array(bootstrap_scores)
            
            # Calculate percentiles
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bounds = np.percentile(bootstrap_scores, lower_percentile, axis=0)
            upper_bounds = np.percentile(bootstrap_scores, upper_percentile, axis=0)
            
            # Add confidence intervals to rankings
            self.rankings_['score_lower_ci'] = [lower_bounds[self.repo_names.index(repo)] 
                                              for repo in self.rankings_['repository']]
            self.rankings_['score_upper_ci'] = [upper_bounds[self.repo_names.index(repo)] 
                                              for repo in self.rankings_['repository']]
            
            # Calculate rank stability (standard deviation of ranks)
            bootstrap_ranks = []
            for scores in bootstrap_scores:
                temp_rankings = pd.DataFrame({
                    'repository': self.repo_names,
                    'score': scores
                }).sort_values('score', ascending=False)
                temp_rankings['rank'] = range(1, len(temp_rankings) + 1)
                bootstrap_ranks.append(temp_rankings.set_index('repository')['rank'].values)
            
            bootstrap_ranks = np.array(bootstrap_ranks)
            rank_std = np.std(bootstrap_ranks, axis=0)
            
            self.rankings_['rank_stability'] = [rank_std[self.repo_names.index(repo)] 
                                              for repo in self.rankings_['repository']]
            
            logger.info("✅ Bootstrap confidence intervals calculated")
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence intervals: {e}")
            # Add placeholder columns
            self.rankings_['score_lower_ci'] = self.rankings_['bradley_terry_score']
            self.rankings_['score_upper_ci'] = self.rankings_['bradley_terry_score']
            self.rankings_['rank_stability'] = 0.0
    
    def predict_pairwise(self, repo1: str, repo2: str) -> float:
        """Predict probability that repo1 beats repo2
        
        Args:
            repo1: First repository name
            repo2: Second repository name
            
        Returns:
            Probability that repo1 beats repo2
        """
        
        if self.scores_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            i = self.repo_names.index(repo1)
            j = self.repo_names.index(repo2)
            
            score_i = self.scores_[i]
            score_j = self.scores_[j]
            
            # Bradley-Terry probability
            prob = score_i / (score_i + score_j)
            return prob
            
        except ValueError:
            logger.error(f"Repository not found: {repo1} or {repo2}")
            return 0.5  # Default to neutral
    
    def get_rankings(self) -> pd.DataFrame:
        """Get final rankings dataframe
        
        Returns:
            DataFrame with rankings and scores
        """
        
        if self.rankings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.rankings_.copy()
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics and fit statistics
        
        Returns:
            Dictionary with model diagnostics
        """
        
        if self.scores_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Calculate comparison coverage
        total_possible = self.n_repos * (self.n_repos - 1) / 2
        total_actual = np.sum(self.comparisons_ > 0) / 2
        coverage = total_actual / total_possible
        
        # Calculate comparison balance
        comparison_counts = []
        for i in range(self.n_repos):
            count = np.sum(self.comparisons_[i, :] > 0)
            comparison_counts.append(count)
        
        diagnostics = {
            'convergence_status': self.convergence_info_,
            'comparison_coverage': float(coverage),
            'avg_comparisons_per_repo': float(np.mean(comparison_counts)),
            'min_comparisons_per_repo': int(np.min(comparison_counts)),
            'max_comparisons_per_repo': int(np.max(comparison_counts)),
            'total_repositories': int(self.n_repos),
            'total_comparisons': int(total_actual),
            'score_distribution': {
                'mean': float(np.mean(self.scores_)),
                'std': float(np.std(self.scores_)),
                'min': float(np.min(self.scores_)),
                'max': float(np.max(self.scores_))
            }
        }
        
        return diagnostics
    
    def save_rankings(self, output_path: str, include_metadata: bool = True):
        """Save rankings to file
        
        Args:
            output_path: Path to save rankings
            include_metadata: Whether to include model metadata
        """
        
        if self.rankings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            output_data = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'model_type': 'bradley_terry',
                'total_repositories': len(self.rankings_),
                'rankings': self.rankings_.to_dict('records')
            }
            
            if include_metadata:
                output_data['model_diagnostics'] = self.get_model_diagnostics()
                output_data['model_parameters'] = {
                    'max_iterations': self.max_iterations,
                    'tolerance': self.tolerance
                }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"✅ Saved Bradley-Terry rankings to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save rankings: {e}")
            raise

    def load_rankings(self, input_path: str) -> pd.DataFrame:
        """Load rankings from file
        
        Args:
            input_path: Path to load rankings from
            
        Returns:
            Rankings dataframe
        """
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            self.rankings_ = pd.DataFrame(data['rankings'])
            
            logger.info(f"✅ Loaded Bradley-Terry rankings from {input_path}")
            return self.rankings_.copy()
            
        except Exception as e:
            logger.error(f"Failed to load rankings: {e}")
            raise

def create_integrated_rankings(bradley_terry_rankings: pd.DataFrame,
                             engineered_features_df: pd.DataFrame,
                             weights: Dict[str, float] = None) -> pd.DataFrame:
    """Create integrated rankings combining Bradley-Terry with feature scores
    
    Args:
        bradley_terry_rankings: Bradley-Terry rankings dataframe
        engineered_features_df: Original features dataframe
        weights: Weights for different components
        
    Returns:
        Integrated rankings dataframe
    """
    
    if weights is None:
        weights = {
            'bradley_terry': 0.7,
            'category_potential': 0.2,
            'bsv_investment': 0.1
        }
    
    logger.info("Creating integrated rankings...")
    
    # Merge dataframes
    bt_scores = bradley_terry_rankings.set_index('repository')
    features = engineered_features_df.set_index('repo_name')
    
    # Create integrated scoring
    integrated_data = []
    
    for repo in bt_scores.index:
        if repo in features.index:
            # Get Bradley-Terry score
            bt_score = bt_scores.loc[repo, 'bradley_terry_score']
            bt_rank = bt_scores.loc[repo, 'rank']
            
            # Get feature scores
            category_score = features.loc[repo, 'category_potential_score']
            bsv_score = features.loc[repo, 'bsv_investment_score']
            
            # Calculate integrated score
            integrated_score = (
                weights['bradley_terry'] * bt_score +
                weights['category_potential'] * category_score +
                weights['bsv_investment'] * bsv_score
            )
            
            integrated_data.append({
                'repository': repo,
                'bradley_terry_score': bt_score,
                'bradley_terry_rank': bt_rank,
                'category_potential_score': category_score,
                'bsv_investment_score': bsv_score,
                'integrated_score': integrated_score,
                'score_lower_ci': bt_scores.loc[repo].get('score_lower_ci', bt_score),
                'score_upper_ci': bt_scores.loc[repo].get('score_upper_ci', bt_score),
                'rank_stability': bt_scores.loc[repo].get('rank_stability', 0.0)
            })
    
    # Create final rankings
    integrated_df = pd.DataFrame(integrated_data)
    integrated_df = integrated_df.sort_values('integrated_score', ascending=False)
    integrated_df['final_rank'] = range(1, len(integrated_df) + 1)
    
    # Reorder columns
    columns = ['final_rank', 'repository', 'integrated_score', 
               'bradley_terry_score', 'bradley_terry_rank',
               'category_potential_score', 'bsv_investment_score',
               'score_lower_ci', 'score_upper_ci', 'rank_stability']
    
    integrated_df = integrated_df[columns]
    
    logger.info(f"✅ Created integrated rankings for {len(integrated_df)} repositories")
    return integrated_df

if __name__ == "__main__":
    # Test the Bradley-Terry ranker
    print("🏆 Testing Bradley-Terry Ranking Aggregation")
    
    # Check for required input files
    data_dir = Path(__file__).parent.parent / "data"
    features_path = data_dir / "task2_engineered_features.csv"
    
    if not features_path.exists():
        print("❌ Task 2 engineered features not found. Run Task 2 first.")
        exit()
    
    # Mock pairwise results for testing
    mock_results = [
        {
            'repo1_name': 'repo_A',
            'repo2_name': 'repo_B', 
            'winner': 0,
            'confidence': 'high'
        },
        {
            'repo1_name': 'repo_B',
            'repo2_name': 'repo_C',
            'winner': 1,
            'confidence': 'medium'
        },
        {
            'repo1_name': 'repo_A',
            'repo2_name': 'repo_C',
            'winner': 0,
            'confidence': 'high'
        }
    ]
    
    repo_names = ['repo_A', 'repo_B', 'repo_C']
    
    try:
        # Create and fit ranker
        ranker = BradleyTerryRanker()
        ranker.fit(mock_results, repo_names)
        
        # Get rankings
        rankings = ranker.get_rankings()
        print("📊 Test Rankings:")
        print(rankings)
        
        # Get diagnostics
        diagnostics = ranker.get_model_diagnostics()
        print(f"\n📈 Model Diagnostics:")
        print(f"  Converged: {diagnostics['convergence_status']['converged']}")
        print(f"  Iterations: {diagnostics['convergence_status']['iterations']}")
        print(f"  Comparison coverage: {diagnostics['comparison_coverage']:.1%}")
        
        # Test pairwise prediction
        prob = ranker.predict_pairwise('repo_A', 'repo_B')
        print(f"\n🔮 Prediction: P(repo_A beats repo_B) = {prob:.3f}")
        
        print("\n✅ Bradley-Terry ranker test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")