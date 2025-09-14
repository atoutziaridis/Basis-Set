"""
BSV Final Scoring and Evaluation System
Implements Task 4: Composite scoring, explainability, and comprehensive evaluation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScoringWeights:
    """Configuration for composite scoring weights"""
    llm_preference: float = 0.60  # Primary signal from LLM pairwise rankings
    technical_execution: float = 0.15  # Velocity, releases, code quality  
    market_adoption: float = 0.15  # Dependents, downloads, stars growth
    team_resilience: float = 0.10  # Bus factor, contributor diversity
    
    def __post_init__(self):
        total = self.llm_preference + self.technical_execution + self.market_adoption + self.team_resilience
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")

@dataclass
class ReasonCode:
    """Structured reason code for explainability"""
    factor: str
    contribution: float
    description: str
    value: Optional[float] = None

@dataclass
class RepositoryScore:
    """Complete scoring result for a repository"""
    repo_name: str
    final_score: float
    component_scores: Dict[str, float]
    reason_codes: List[ReasonCode]
    funding_gate_multiplier: float
    rank: int
    
class FinalScorer:
    """
    BSV Final Scoring and Evaluation System
    
    Combines all signals into interpretable final rankings with comprehensive evaluation.
    Implements Task 4.1: Composite Scoring Framework
    """
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        """Initialize scorer with configurable weights"""
        self.weights = weights or ScoringWeights()
        self.feature_mappings = self._define_feature_mappings()
        self.results: List[RepositoryScore] = []
        
        logger.info(f"FinalScorer initialized with weights: LLM={self.weights.llm_preference:.1%}, "
                   f"Technical={self.weights.technical_execution:.1%}, "
                   f"Market={self.weights.market_adoption:.1%}, "
                   f"Team={self.weights.team_resilience:.1%}")
    
    def _define_feature_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Define which features contribute to each scoring component"""
        return {
            'technical_execution': {
                'velocity': ['commit_velocity_score', 'commits_6_months', 'avg_commits_per_week'],
                'releases': ['release_cadence_score', 'total_releases', 'releases_last_year'],
                'code_quality': ['code_quality_score', 'has_ci_cd', 'has_tests', 'config_completeness_score']
            },
            'market_adoption': {
                'dependents': ['dependents_count', 'network_count'],
                'downloads': ['pypi_downloads', 'npm_downloads', 'cargo_downloads'],
                'growth': ['stars_per_month', 'growth_trajectory_score', 'engagement_score']
            },
            'team_resilience': {
                'bus_factor': ['bus_factor', 'top_contributor_percentage'],
                'diversity': ['contribution_gini', 'active_contributors', 'total_contributors'],
                'community': ['community_health_score', 'team_resilience_score']
            }
        }
    
    def load_data(self, task2_path: str, task3_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Task 2 engineered features and Task 3 LLM rankings"""
        logger.info("Loading Task 2 and Task 3 results...")
        
        # Load Task 2 features
        task2_df = pd.read_csv(task2_path)
        logger.info(f"Loaded {len(task2_df)} repositories with {len(task2_df.columns)} features")
        
        # Load Task 3 LLM rankings
        task3_df = pd.read_csv(task3_path)
        logger.info(f"Loaded {len(task3_df)} LLM rankings")
        
        return task2_df, task3_df
    
    def _normalize_features(self, df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        """Normalize features to [0,1] range using min-max scaling"""
        df_norm = df.copy()
        
        for feature in feature_list:
            if feature in df.columns:
                col = df[feature]
                if col.dtype in ['int64', 'float64'] and col.notna().any():
                    col_min, col_max = col.min(), col.max()
                    if col_max > col_min:  # Avoid division by zero
                        df_norm[feature] = (col - col_min) / (col_max - col_min)
                    else:
                        df_norm[feature] = 0.5  # Set to middle value if no variation
                else:
                    df_norm[feature] = 0.0  # Set non-numeric to 0
        
        return df_norm
    
    def _calculate_component_score(self, df: pd.DataFrame, component: str, 
                                 repo_idx: int) -> Tuple[float, List[ReasonCode]]:
        """Calculate score for a specific component (technical_execution, market_adoption, team_resilience)"""
        if component not in self.feature_mappings:
            return 0.0, []
        
        component_features = self.feature_mappings[component]
        scores = []
        reason_codes = []
        
        for subcategory, features in component_features.items():
            # Calculate subcategory score as average of available features
            subcategory_scores = []
            for feature in features:
                if feature in df.columns:
                    value = df.iloc[repo_idx][feature]
                    if pd.notna(value) and isinstance(value, (int, float)):
                        subcategory_scores.append(float(value))
            
            if subcategory_scores:
                subcategory_score = np.mean(subcategory_scores)
                scores.append(subcategory_score)
                
                # Add reason code for significant contributions (>0.7)
                if subcategory_score > 0.7:
                    reason_codes.append(ReasonCode(
                        factor=f"{component}_{subcategory}",
                        contribution=subcategory_score,
                        description=f"Strong {subcategory.replace('_', ' ')} performance",
                        value=subcategory_score
                    ))
        
        final_score = np.mean(scores) if scores else 0.0
        return final_score, reason_codes
    
    def _calculate_funding_gate(self, df: pd.DataFrame, repo_idx: int) -> float:
        """Calculate funding gate multiplier: max(0.6, 1 - p_institutional_funding)"""
        # Use funding_confidence as proxy for institutional funding probability
        funding_confidence = df.iloc[repo_idx].get('funding_confidence', 0.0)
        
        # Convert funding confidence to institutional funding probability
        # Higher confidence = higher probability of institutional funding
        p_institutional = funding_confidence
        
        # Apply funding gate formula
        multiplier = max(0.6, 1.0 - p_institutional)
        return multiplier
    
    def _generate_reason_codes(self, repo_scores: Dict[str, float], 
                             component_reasons: List[ReasonCode],
                             repo_data: pd.Series) -> List[ReasonCode]:
        """Generate top 3 reason codes explaining the ranking"""
        all_reasons = component_reasons.copy()
        
        # Add high-level component reasons
        for component, score in repo_scores.items():
            if score > 0.7:
                if component == 'llm_preference':
                    all_reasons.append(ReasonCode(
                        factor="llm_preference",
                        contribution=score,
                        description="High LLM preference score indicates strong innovation potential",
                        value=score
                    ))
                elif component == 'technical_execution':
                    all_reasons.append(ReasonCode(
                        factor="technical_execution", 
                        contribution=score,
                        description="Excellent development velocity and code quality",
                        value=score
                    ))
                elif component == 'market_adoption':
                    all_reasons.append(ReasonCode(
                        factor="market_adoption",
                        contribution=score,
                        description="Strong community adoption and growth signals",
                        value=score
                    ))
                elif component == 'team_resilience':
                    all_reasons.append(ReasonCode(
                        factor="team_resilience",
                        contribution=score,
                        description="Healthy contributor diversity and team structure",
                        value=score
                    ))
        
        # Add funding advantage if applicable
        funding_risk = repo_data.get('funding_risk_level', 'unknown')
        if funding_risk in ['low_risk_unfunded', 'unfunded']:
            all_reasons.append(ReasonCode(
                factor="funding_advantage",
                contribution=0.8,
                description="No institutional funding detected - higher investment potential",
                value=1.0
            ))
        
        # Sort by contribution and return top 3
        all_reasons.sort(key=lambda x: x.contribution, reverse=True)
        return all_reasons[:3]
    
    def calculate_final_scores(self, task2_df: pd.DataFrame, task3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final composite scores combining all signals
        
        Score Components:
        - LLM Preference Score: 60% (primary signal)
        - Technical Execution: 15% (velocity, releases, code quality) 
        - Market Adoption: 15% (dependents, downloads, stars growth)
        - Team Resilience: 10% (bus factor, contributor diversity)
        - Funding Gate: Multiply by max(0.6, 1 - p_institutional_funding)
        """
        logger.info("Calculating final composite scores...")
        
        # Merge datasets on repository identifier
        # Use repo_name or full_name as key
        if 'repo_name' in task2_df.columns and 'repository' in task3_df.columns:
            merged_df = task2_df.merge(task3_df, left_on='repo_name', right_on='repository', how='inner')
        elif 'full_name' in task2_df.columns and 'repository' in task3_df.columns:
            merged_df = task2_df.merge(task3_df, left_on='full_name', right_on='repository', how='inner')
        else:
            # Fallback: try to match by position (assumes same order)
            logger.warning("No clear key for merging - using positional matching")
            task3_df_indexed = task3_df.copy()
            task3_df_indexed.index = task2_df.index[:len(task3_df)]
            merged_df = pd.concat([task2_df, task3_df_indexed], axis=1)
        
        logger.info(f"Successfully merged {len(merged_df)} repositories")
        
        # Normalize features for consistent scoring
        numeric_columns = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        merged_df_norm = self._normalize_features(merged_df, numeric_columns)
        
        # Calculate scores for each repository
        results = []
        
        for idx in range(len(merged_df_norm)):
            repo_data = merged_df.iloc[idx]
            repo_name = repo_data.get('repo_name', repo_data.get('repository', f'repo_{idx}'))
            
            # 1. LLM Preference Score (60%)
            llm_score = merged_df_norm.iloc[idx].get('integrated_score', 
                       merged_df_norm.iloc[idx].get('bradley_terry_score', 0.5))
            
            # 2. Technical Execution Score (15%)
            tech_score, tech_reasons = self._calculate_component_score(
                merged_df_norm, 'technical_execution', idx)
            
            # 3. Market Adoption Score (15%) 
            market_score, market_reasons = self._calculate_component_score(
                merged_df_norm, 'market_adoption', idx)
            
            # 4. Team Resilience Score (10%)
            team_score, team_reasons = self._calculate_component_score(
                merged_df_norm, 'team_resilience', idx)
            
            # Combine component scores
            component_scores = {
                'llm_preference': float(llm_score),
                'technical_execution': float(tech_score),
                'market_adoption': float(market_score), 
                'team_resilience': float(team_score)
            }
            
            # Calculate weighted composite score
            composite_score = (
                self.weights.llm_preference * component_scores['llm_preference'] +
                self.weights.technical_execution * component_scores['technical_execution'] +
                self.weights.market_adoption * component_scores['market_adoption'] + 
                self.weights.team_resilience * component_scores['team_resilience']
            )
            
            # Apply funding gate
            funding_multiplier = self._calculate_funding_gate(merged_df, idx)
            final_score = composite_score * funding_multiplier
            
            # Ensure final score is in [0,1] range
            final_score = np.clip(final_score, 0.0, 1.0)
            
            # Generate reason codes
            all_reasons = tech_reasons + market_reasons + team_reasons
            reason_codes = self._generate_reason_codes(component_scores, all_reasons, repo_data)
            
            # Create repository score result
            repo_result = RepositoryScore(
                repo_name=repo_name,
                final_score=final_score,
                component_scores=component_scores,
                reason_codes=reason_codes,
                funding_gate_multiplier=funding_multiplier,
                rank=0  # Will be set after sorting
            )
            
            results.append(repo_result)
        
        # Sort by final score and assign ranks
        results.sort(key=lambda x: x.final_score, reverse=True)
        for i, result in enumerate(results, 1):
            result.rank = i
        
        self.results = results
        
        # Convert to DataFrame for output
        output_data = []
        for result in results:
            row = {
                'repo_name': result.repo_name,
                'rank': result.rank,
                'final_score': result.final_score,
                'llm_preference_score': result.component_scores['llm_preference'],
                'technical_execution_score': result.component_scores['technical_execution'],
                'market_adoption_score': result.component_scores['market_adoption'],
                'team_resilience_score': result.component_scores['team_resilience'],
                'funding_gate_multiplier': result.funding_gate_multiplier,
                'reason_1': result.reason_codes[0].description if len(result.reason_codes) > 0 else "",
                'reason_2': result.reason_codes[1].description if len(result.reason_codes) > 1 else "",
                'reason_3': result.reason_codes[2].description if len(result.reason_codes) > 2 else "",
                'reason_1_factor': result.reason_codes[0].factor if len(result.reason_codes) > 0 else "",
                'reason_2_factor': result.reason_codes[1].factor if len(result.reason_codes) > 1 else "",
                'reason_3_factor': result.reason_codes[2].factor if len(result.reason_codes) > 2 else "",
                'reason_1_value': result.reason_codes[0].value if len(result.reason_codes) > 0 else None,
                'reason_2_value': result.reason_codes[1].value if len(result.reason_codes) > 1 else None,
                'reason_3_value': result.reason_codes[2].value if len(result.reason_codes) > 2 else None,
            }
            
            # Add original data for reference
            original_data = merged_df[merged_df.get('repo_name', merged_df.get('repository', '')) == result.repo_name].iloc[0] if not merged_df.empty else {}
            row.update({
                'stars': original_data.get('stars', 0),
                'forks': original_data.get('forks', 0),
                'created_at': original_data.get('created_at', ''),
                'funding_risk_level': original_data.get('funding_risk_level', 'unknown'),
                'category_potential_score': original_data.get('category_potential_score', 0),
                'bsv_investment_score': original_data.get('bsv_investment_score', 0)
            })
            
            output_data.append(row)
        
        result_df = pd.DataFrame(output_data)
        
        logger.info(f"Calculated final scores for {len(result_df)} repositories")
        logger.info(f"Score range: {result_df['final_score'].min():.3f} - {result_df['final_score'].max():.3f}")
        
        return result_df
    
    def save_results(self, results_df: pd.DataFrame, output_path: str, 
                    metadata: Optional[Dict[str, Any]] = None):
        """Save final scoring results with metadata"""
        
        # Save main results CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"Final scoring results saved to {output_path}")
        
        # Save detailed metadata
        metadata_path = output_path.replace('.csv', '_metadata.json')
        full_metadata = {
            'generated_at': datetime.now().isoformat(),
            'scoring_weights': asdict(self.weights),
            'total_repositories': len(results_df),
            'score_statistics': {
                'mean': float(results_df['final_score'].mean()),
                'std': float(results_df['final_score'].std()),
                'min': float(results_df['final_score'].min()),
                'max': float(results_df['final_score'].max()),
                'median': float(results_df['final_score'].median())
            },
            'component_score_ranges': {
                'llm_preference': {
                    'min': float(results_df['llm_preference_score'].min()),
                    'max': float(results_df['llm_preference_score'].max()),
                    'mean': float(results_df['llm_preference_score'].mean())
                },
                'technical_execution': {
                    'min': float(results_df['technical_execution_score'].min()),
                    'max': float(results_df['technical_execution_score'].max()),
                    'mean': float(results_df['technical_execution_score'].mean())
                },
                'market_adoption': {
                    'min': float(results_df['market_adoption_score'].min()),
                    'max': float(results_df['market_adoption_score'].max()),
                    'mean': float(results_df['market_adoption_score'].mean())
                },
                'team_resilience': {
                    'min': float(results_df['team_resilience_score'].min()),
                    'max': float(results_df['team_resilience_score'].max()),
                    'mean': float(results_df['team_resilience_score'].mean())
                }
            },
            'top_10_repositories': results_df.head(10)[['repo_name', 'final_score', 'reason_1']].to_dict('records')
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return metadata_path

def main():
    """Main execution function for Task 4.1: Composite Scoring Framework"""
    logger.info("Starting BSV Final Scoring System - Task 4.1")
    
    # Initialize scorer with default weights
    scorer = FinalScorer()
    
    # Define input paths
    project_root = Path(__file__).parent.parent
    task2_path = project_root / "data" / "test_task3_dataset.csv"  # Contains Task 2 features for Task 3 repositories
    task3_path = project_root / "data" / "task3_final_llm_rankings.csv"
    
    # Check if input files exist
    if not task2_path.exists():
        logger.error(f"Task 2 results not found: {task2_path}")
        return
    
    if not task3_path.exists():
        logger.error(f"Task 3 results not found: {task3_path}")
        return
    
    try:
        # Load data
        task2_df, task3_df = scorer.load_data(str(task2_path), str(task3_path))
        
        # Calculate final scores
        results_df = scorer.calculate_final_scores(task2_df, task3_df)
        
        # Save results
        output_path = project_root / "data" / "task4_final_scores.csv"
        metadata_path = scorer.save_results(results_df, str(output_path))
        
        # Display summary
        print("\n" + "="*60)
        print("🎉 TASK 4.1 COMPOSITE SCORING COMPLETE")
        print("="*60)
        print(f"📊 Repositories scored: {len(results_df)}")
        print(f"🏆 Score range: {results_df['final_score'].min():.3f} - {results_df['final_score'].max():.3f}")
        print(f"📈 Mean score: {results_df['final_score'].mean():.3f}")
        print()
        print("📁 Output files:")
        print(f"   • Final scores: {output_path}")
        print(f"   • Metadata: {metadata_path}")
        print()
        print("🏆 Top 10 Repositories:")
        for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['repo_name']:<25} | Score: {row['final_score']:.3f} | {row['reason_1']}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Final scoring failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
