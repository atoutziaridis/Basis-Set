"""
Strategic Pair Selector - Task 3.2
Intelligent sampling strategy for pairwise repository comparisons
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
import random
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategicPairSelector:
    """Strategic selection of repository pairs for LLM comparison"""
    
    def __init__(self, target_comparisons: int = 300):
        """Initialize with target number of comparisons"""
        self.target_comparisons = target_comparisons
        self.selected_pairs = []
        self.coverage_stats = {}
        
    def select_pairs(self, df: pd.DataFrame, score_column: str = 'category_potential_score') -> List[Tuple[int, int]]:
        """
        Select strategic pairs for comparison
        
        Args:
            df: DataFrame with repository data
            score_column: Column to use for preliminary scoring
            
        Returns:
            List of (index1, index2) pairs
        """
        
        logger.info(f"Selecting {self.target_comparisons} strategic pairs from {len(df)} repositories...")
        
        n_repos = len(df)
        if n_repos < 2:
            logger.warning("Need at least 2 repositories for pairwise comparison")
            return []
        
        # Initialize selected pairs set to avoid duplicates
        selected_pairs_set = set()
        pairs_list = []
        
        # Strategy 1: Random baseline pairs (20% of target)
        baseline_count = max(int(self.target_comparisons * 0.2), min(50, n_repos * 2))
        baseline_pairs = self._select_random_pairs(n_repos, baseline_count)
        pairs_list.extend(baseline_pairs)
        selected_pairs_set.update(baseline_pairs)
        
        # Strategy 2: Uncertainty/similarity-based pairs (40% of target)
        uncertainty_count = int(self.target_comparisons * 0.4)
        uncertainty_pairs = self._select_uncertainty_pairs(df, score_column, uncertainty_count, selected_pairs_set)
        pairs_list.extend(uncertainty_pairs)
        selected_pairs_set.update(uncertainty_pairs)
        
        # Strategy 3: Coverage optimization (25% of target)
        coverage_count = int(self.target_comparisons * 0.25)
        coverage_pairs = self._select_coverage_pairs(n_repos, coverage_count, selected_pairs_set)
        pairs_list.extend(coverage_pairs)
        selected_pairs_set.update(coverage_pairs)
        
        # Strategy 4: Quality-based pairs (15% of target)
        quality_count = self.target_comparisons - len(pairs_list)
        quality_pairs = self._select_quality_pairs(df, score_column, quality_count, selected_pairs_set)
        pairs_list.extend(quality_pairs)
        
        # Final validation and statistics
        self.selected_pairs = pairs_list
        self._calculate_coverage_stats(n_repos)
        
        logger.info(f"✅ Selected {len(self.selected_pairs)} pairs with {self.coverage_stats['avg_appearances']:.1f} avg appearances per repo")
        
        return self.selected_pairs
    
    def _select_random_pairs(self, n_repos: int, count: int) -> List[Tuple[int, int]]:
        """Select random baseline pairs"""
        
        pairs = []
        max_attempts = count * 3  # Avoid infinite loops
        attempts = 0
        
        while len(pairs) < count and attempts < max_attempts:
            i, j = random.sample(range(n_repos), 2)
            pair = tuple(sorted([i, j]))
            
            if pair not in pairs:
                pairs.append(pair)
            
            attempts += 1
        
        logger.info(f"Selected {len(pairs)} random baseline pairs")
        return pairs
    
    def _select_uncertainty_pairs(self, df: pd.DataFrame, score_column: str, count: int, 
                                 existing_pairs: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Select pairs where scores are similar (high uncertainty)"""
        
        if score_column not in df.columns:
            logger.warning(f"Score column {score_column} not found, using random selection")
            return self._select_random_pairs(len(df), count)
        
        scores = df[score_column].fillna(0).values
        n_repos = len(scores)
        
        # Calculate pairwise score differences
        pair_scores = []
        for i in range(n_repos):
            for j in range(i + 1, n_repos):
                pair = tuple(sorted([i, j]))
                if pair not in existing_pairs:
                    score_diff = abs(scores[i] - scores[j])
                    pair_scores.append((score_diff, pair))
        
        # Sort by smallest differences (most uncertain/similar)
        pair_scores.sort(key=lambda x: x[0])
        
        # Select pairs with smallest score differences
        selected_pairs = [pair for _, pair in pair_scores[:count]]
        
        logger.info(f"Selected {len(selected_pairs)} uncertainty-based pairs (avg score diff: {np.mean([score for score, _ in pair_scores[:count]]):.3f})")
        
        return selected_pairs
    
    def _select_coverage_pairs(self, n_repos: int, count: int, 
                              existing_pairs: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Select pairs to ensure good coverage of all repositories"""
        
        # Count current appearances for each repository
        appearance_count = [0] * n_repos
        for i, j in existing_pairs:
            appearance_count[i] += 1
            appearance_count[j] += 1
        
        pairs = []
        max_attempts = count * 5
        attempts = 0
        
        while len(pairs) < count and attempts < max_attempts:
            # Prioritize repositories with fewer appearances
            weights = [1.0 / (appearances + 1) for appearances in appearance_count]
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Select first repo based on coverage need
            i = np.random.choice(n_repos, p=weights)
            
            # Select second repo (avoid same, prioritize low coverage)
            remaining_weights = weights.copy()
            remaining_weights[i] = 0
            if remaining_weights.sum() > 0:
                remaining_weights = remaining_weights / remaining_weights.sum()
                j = np.random.choice(n_repos, p=remaining_weights)
            else:
                j = random.choice([x for x in range(n_repos) if x != i])
            
            pair = tuple(sorted([i, j]))
            
            if pair not in existing_pairs and pair not in pairs:
                pairs.append(pair)
                appearance_count[i] += 1
                appearance_count[j] += 1
            
            attempts += 1
        
        logger.info(f"Selected {len(pairs)} coverage-optimized pairs")
        return pairs
    
    def _select_quality_pairs(self, df: pd.DataFrame, score_column: str, count: int,
                             existing_pairs: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Select pairs comparing high-quality repositories"""
        
        if score_column not in df.columns or count <= 0:
            return []
        
        scores = df[score_column].fillna(0).values
        n_repos = len(scores)
        
        # Get top repositories by score
        top_repo_indices = np.argsort(scores)[-min(20, n_repos):]  # Top 20 or all if fewer
        
        pairs = []
        max_attempts = count * 3
        attempts = 0
        
        while len(pairs) < count and attempts < max_attempts:
            # Select from top repositories
            i, j = random.sample(list(top_repo_indices), 2)
            pair = tuple(sorted([i, j]))
            
            if pair not in existing_pairs and pair not in pairs:
                pairs.append(pair)
            
            attempts += 1
        
        logger.info(f"Selected {len(pairs)} quality-focused pairs from top repositories")
        return pairs
    
    def _calculate_coverage_stats(self, n_repos: int):
        """Calculate coverage statistics for selected pairs"""
        
        appearance_count = [0] * n_repos
        for i, j in self.selected_pairs:
            appearance_count[i] += 1
            appearance_count[j] += 1
        
        self.coverage_stats = {
            'total_pairs': len(self.selected_pairs),
            'total_repos': n_repos,
            'avg_appearances': float(np.mean(appearance_count)),
            'min_appearances': int(np.min(appearance_count)),
            'max_appearances': int(np.max(appearance_count)),
            'repos_without_pairs': sum(1 for count in appearance_count if count == 0),
            'coverage_percentage': float((1 - sum(1 for count in appearance_count if count == 0) / n_repos) * 100)
        }
    
    def get_coverage_report(self) -> Dict:
        """Get detailed coverage report"""
        return {
            'pair_selection_summary': self.coverage_stats,
            'strategy_breakdown': {
                'random_baseline': f"~{int(self.target_comparisons * 0.2)} pairs (20%)",
                'uncertainty_focused': f"~{int(self.target_comparisons * 0.4)} pairs (40%)",
                'coverage_optimized': f"~{int(self.target_comparisons * 0.25)} pairs (25%)",
                'quality_focused': f"~{int(self.target_comparisons * 0.15)} pairs (15%)"
            },
            'quality_metrics': {
                'repository_coverage': f"{self.coverage_stats['coverage_percentage']:.1f}%",
                'avg_comparisons_per_repo': f"{self.coverage_stats['avg_appearances']:.1f}",
                'comparison_efficiency': f"{len(self.selected_pairs)} pairs for {self.coverage_stats['total_repos']} repos"
            }
        }
    
    def save_pairs(self, pairs: List[Tuple[int, int]], output_path: str, df: pd.DataFrame = None):
        """Save selected pairs with metadata"""
        
        try:
            # Prepare pair data
            pair_data = []
            for i, (idx1, idx2) in enumerate(pairs):
                pair_info = {
                    'pair_id': i,
                    'repo1_index': int(idx1),
                    'repo2_index': int(idx2)
                }
                
                # Add repository names if DataFrame provided
                if df is not None:
                    pair_info['repo1_name'] = df.iloc[idx1].get('repo_name', f'repo_{idx1}')
                    pair_info['repo2_name'] = df.iloc[idx2].get('repo_name', f'repo_{idx2}')
                    
                    # Add preliminary scores for analysis
                    if 'category_potential_score' in df.columns:
                        pair_info['repo1_score'] = float(df.iloc[idx1]['category_potential_score'])
                        pair_info['repo2_score'] = float(df.iloc[idx2]['category_potential_score'])
                        pair_info['score_difference'] = abs(pair_info['repo1_score'] - pair_info['repo2_score'])
                
                pair_data.append(pair_info)
            
            # Complete output data
            output_data = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'total_pairs': len(pairs),
                'selection_strategy': 'strategic_mixed',
                'coverage_stats': self.coverage_stats,
                'pairs': pair_data
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"✅ Saved {len(pairs)} pairs to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save pairs: {e}")
            raise
    
    def load_pairs(self, input_path: str) -> List[Tuple[int, int]]:
        """Load pairs from saved file"""
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            pairs = [(pair['repo1_index'], pair['repo2_index']) for pair in data['pairs']]
            
            logger.info(f"✅ Loaded {len(pairs)} pairs from {input_path}")
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to load pairs: {e}")
            raise

if __name__ == "__main__":
    # Test the pair selector
    test_data_path = Path(__file__).parent.parent / "data" / "task2_engineered_features.csv"
    
    if test_data_path.exists():
        # Load test data
        df = pd.read_csv(test_data_path)
        
        # Create selector
        selector = StrategicPairSelector(target_comparisons=20)  # Small number for testing
        
        # Select pairs
        pairs = selector.select_pairs(df, 'category_potential_score')
        
        # Save pairs
        output_path = Path(__file__).parent.parent / "data" / "selected_pairs.json"
        selector.save_pairs(pairs, str(output_path), df)
        
        # Show coverage report
        report = selector.get_coverage_report()
        print(f"📊 Pair Selection Report:")
        print("=" * 40)
        print(f"Total pairs selected: {report['pair_selection_summary']['total_pairs']}")
        print(f"Repository coverage: {report['quality_metrics']['repository_coverage']}")
        print(f"Avg comparisons per repo: {report['quality_metrics']['avg_comparisons_per_repo']}")
        print()
        print("Strategy breakdown:")
        for strategy, count in report['strategy_breakdown'].items():
            print(f"  • {strategy}: {count}")
        
        print(f"\n✅ Pair selection complete! Results saved to {output_path}")
        
    else:
        print("❌ Task 2 engineered features not found. Run Task 2 first.")