"""
BSV Explainability and Reasoning System
Implements Task 4.2: Feature contribution analysis and SHAP values
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

# Try to import SHAP, install if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FeatureContribution:
    """Individual feature contribution to final score"""
    feature_name: str
    contribution: float
    normalized_contribution: float
    feature_value: Union[float, str, bool]
    percentile_rank: float
    importance_category: str  # 'high', 'medium', 'low'

@dataclass
class RepositoryExplanation:
    """Complete explanation for a repository's ranking"""
    repo_name: str
    rank: int
    final_score: float
    component_scores: Dict[str, float]
    top_positive_features: List[FeatureContribution]
    top_negative_features: List[FeatureContribution]
    shap_values: Optional[Dict[str, float]]
    human_readable_summary: str
    comparative_advantages: List[str]
    areas_for_improvement: List[str]

class ExplainabilityAnalyzer:
    """
    BSV Explainability and Reasoning System
    
    Provides detailed feature contribution analysis and human-readable explanations
    for repository rankings. Implements Task 4.2: Explainability and Reasoning.
    """
    
    def __init__(self):
        """Initialize explainability analyzer"""
        self.feature_categories = self._define_feature_categories()
        self.explanations: List[RepositoryExplanation] = []
        
        logger.info("ExplainabilityAnalyzer initialized")
    
    def _define_feature_categories(self) -> Dict[str, List[str]]:
        """Define feature categories for structured analysis"""
        return {
            'execution_velocity': [
                'commit_velocity_score', 'commits_6_months', 'avg_commits_per_week',
                'release_cadence_score', 'total_releases', 'releases_last_year',
                'maintenance_activity_score', 'development_consistency_score'
            ],
            'code_quality': [
                'code_quality_score', 'has_ci_cd', 'has_tests', 'config_completeness_score',
                'operational_readiness_score', 'api_stability_score', 'documentation_score'
            ],
            'community_adoption': [
                'stars', 'forks', 'watchers', 'stars_per_month', 'engagement_score',
                'growth_trajectory_score', 'network_effects_score'
            ],
            'market_signals': [
                'dependents_count', 'pypi_downloads', 'npm_downloads', 'cargo_downloads',
                'has_package', 'subscribers_count', 'network_count'
            ],
            'team_health': [
                'bus_factor', 'total_contributors', 'active_contributors',
                'contribution_gini', 'team_resilience_score', 'community_health_score'
            ],
            'innovation_potential': [
                'category_potential_score', 'problem_ambition_score', 'technology_differentiation_score',
                'commercial_viability_score', 'market_readiness_score'
            ],
            'funding_risk': [
                'funding_confidence', 'funding_risk_level', 'total_funding_indicators'
            ]
        }
    
    def load_scored_data(self, scores_path: str, features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load final scores and feature data"""
        logger.info("Loading scored data and features...")
        
        scores_df = pd.read_csv(scores_path)
        features_df = pd.read_csv(features_path)
        
        logger.info(f"Loaded {len(scores_df)} scored repositories")
        logger.info(f"Loaded {len(features_df)} repositories with {len(features_df.columns)} features")
        
        return scores_df, features_df
    
    def _calculate_feature_contributions(self, features_df: pd.DataFrame, 
                                       target_scores: pd.Series) -> pd.DataFrame:
        """Calculate feature contributions using correlation analysis"""
        
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Calculate correlations with final scores
        correlations = {}
        for feature in numeric_features.columns:
            if feature not in ['rank']:  # Exclude rank as it's inverse correlated
                corr = numeric_features[feature].corr(target_scores)
                if pd.notna(corr):
                    correlations[feature] = corr
        
        # Convert to DataFrame for easier handling
        contrib_df = pd.DataFrame([
            {'feature': k, 'correlation': v, 'abs_correlation': abs(v)}
            for k, v in correlations.items()
        ]).sort_values('abs_correlation', ascending=False)
        
        return contrib_df
    
    def _calculate_shap_values(self, features_df: pd.DataFrame, 
                             target_scores: pd.Series) -> Optional[Dict[str, np.ndarray]]:
        """Calculate SHAP values if SHAP is available"""
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - skipping SHAP analysis")
            return None
        
        try:
            # Prepare data for SHAP
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            # Remove features with no variance
            numeric_features = numeric_features.loc[:, numeric_features.std() > 0]
            
            # Simple linear model for SHAP analysis
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            # Handle missing values
            X = numeric_features.fillna(numeric_features.mean())
            y = target_scores
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Calculate SHAP values using Linear explainer
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            
            # Return SHAP values as dictionary
            shap_dict = {}
            for i, feature in enumerate(X.columns):
                shap_dict[feature] = shap_values[:, i]
            
            logger.info("SHAP values calculated successfully")
            return shap_dict
            
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            return None
    
    def _get_feature_percentile(self, feature_value: float, feature_series: pd.Series) -> float:
        """Calculate percentile rank of a feature value"""
        if pd.isna(feature_value):
            return 50.0  # Default to median
        
        percentile = (feature_series <= feature_value).mean() * 100
        return percentile
    
    def _categorize_importance(self, contribution: float, threshold_high: float = 0.1, 
                             threshold_medium: float = 0.05) -> str:
        """Categorize feature importance as high/medium/low"""
        abs_contrib = abs(contribution)
        if abs_contrib >= threshold_high:
            return 'high'
        elif abs_contrib >= threshold_medium:
            return 'medium'
        else:
            return 'low'
    
    def _generate_human_readable_summary(self, repo_explanation: RepositoryExplanation) -> str:
        """Generate human-readable explanation summary"""
        
        repo_name = repo_explanation.repo_name
        rank = repo_explanation.rank
        score = repo_explanation.final_score
        
        # Start with ranking context
        summary_parts = []
        
        if rank == 1:
            summary_parts.append(f"{repo_name} ranks #1 with a final score of {score:.3f}.")
        elif rank <= 3:
            summary_parts.append(f"{repo_name} ranks #{rank} with a strong score of {score:.3f}.")
        elif rank <= 5:
            summary_parts.append(f"{repo_name} ranks #{rank} with a moderate score of {score:.3f}.")
        else:
            summary_parts.append(f"{repo_name} ranks #{rank} with a score of {score:.3f}.")
        
        # Add component score insights
        components = repo_explanation.component_scores
        
        # Find strongest component
        strongest_component = max(components.items(), key=lambda x: x[1])
        if strongest_component[1] > 0.7:
            component_name = strongest_component[0].replace('_', ' ').title()
            summary_parts.append(f"Its strongest aspect is {component_name} ({strongest_component[1]:.2f}).")
        
        # Add top positive features
        if repo_explanation.top_positive_features:
            top_feature = repo_explanation.top_positive_features[0]
            if top_feature.importance_category == 'high':
                summary_parts.append(f"Key strength: {top_feature.feature_name.replace('_', ' ')} "
                                   f"(ranks in {top_feature.percentile_rank:.0f}th percentile).")
        
        # Add areas for improvement if any
        if repo_explanation.areas_for_improvement:
            improvement = repo_explanation.areas_for_improvement[0]
            summary_parts.append(f"Primary improvement area: {improvement}.")
        
        return " ".join(summary_parts)
    
    def _generate_comparative_advantages(self, repo_data: pd.Series, features_df: pd.DataFrame,
                                       top_features: List[FeatureContribution]) -> List[str]:
        """Generate comparative advantages vs other repositories"""
        
        advantages = []
        
        for feature in top_features[:3]:  # Top 3 features
            if feature.percentile_rank >= 80:  # Top 20%
                feature_readable = feature.feature_name.replace('_', ' ').title()
                advantages.append(f"Top 20% in {feature_readable}")
            elif feature.percentile_rank >= 60:  # Top 40%
                feature_readable = feature.feature_name.replace('_', ' ').title()
                advantages.append(f"Above average in {feature_readable}")
        
        return advantages[:3]  # Limit to top 3 advantages
    
    def _generate_improvement_areas(self, repo_data: pd.Series, features_df: pd.DataFrame,
                                  bottom_features: List[FeatureContribution]) -> List[str]:
        """Generate areas for improvement"""
        
        improvements = []
        
        for feature in bottom_features[:3]:  # Bottom 3 features
            if feature.percentile_rank <= 20:  # Bottom 20%
                feature_readable = feature.feature_name.replace('_', ' ').title()
                improvements.append(f"Improve {feature_readable}")
        
        return improvements[:2]  # Limit to top 2 improvement areas
    
    def analyze_repository_explanations(self, scores_df: pd.DataFrame, 
                                      features_df: pd.DataFrame) -> List[RepositoryExplanation]:
        """Generate comprehensive explanations for all repositories"""
        
        logger.info("Generating repository explanations...")
        
        # Merge scores with features
        if 'repo_name' in features_df.columns:
            merged_df = scores_df.merge(features_df, on='repo_name', how='left')
        else:
            # Fallback to positional matching
            merged_df = pd.concat([scores_df, features_df], axis=1)
        
        # Calculate feature contributions
        contribution_df = self._calculate_feature_contributions(features_df, scores_df['final_score'])
        
        # Calculate SHAP values
        shap_values = self._calculate_shap_values(features_df, scores_df['final_score'])
        
        explanations = []
        
        for idx, row in scores_df.iterrows():
            repo_name = row['repo_name']
            
            # Get repository data
            if 'repo_name' in features_df.columns:
                repo_features = features_df[features_df['repo_name'] == repo_name].iloc[0]
            else:
                repo_features = features_df.iloc[idx]
            
            # Calculate individual feature contributions for this repository
            feature_contributions = []
            
            for _, contrib_row in contribution_df.iterrows():
                feature_name = contrib_row['feature']
                correlation = contrib_row['correlation']
                
                if feature_name in repo_features.index:
                    feature_value = repo_features[feature_name]
                    
                    # Calculate normalized contribution (correlation * normalized feature value)
                    if pd.notna(feature_value) and isinstance(feature_value, (int, float)):
                        # Normalize feature value to [0,1] based on dataset range
                        feature_series = features_df[feature_name]
                        if feature_series.max() > feature_series.min():
                            normalized_value = (feature_value - feature_series.min()) / (feature_series.max() - feature_series.min())
                        else:
                            normalized_value = 0.5
                        
                        # Calculate contribution
                        contribution = correlation * normalized_value
                        
                        # Get percentile rank
                        percentile = self._get_feature_percentile(feature_value, feature_series)
                        
                        # Categorize importance
                        importance = self._categorize_importance(contribution)
                        
                        feature_contrib = FeatureContribution(
                            feature_name=feature_name,
                            contribution=contribution,
                            normalized_contribution=contribution,
                            feature_value=feature_value,
                            percentile_rank=percentile,
                            importance_category=importance
                        )
                        
                        feature_contributions.append(feature_contrib)
            
            # Sort by contribution
            feature_contributions.sort(key=lambda x: x.contribution, reverse=True)
            
            # Split into positive and negative contributions
            positive_features = [f for f in feature_contributions if f.contribution > 0][:5]
            negative_features = [f for f in feature_contributions if f.contribution < 0][:3]
            
            # Get SHAP values for this repository
            repo_shap_values = None
            if shap_values:
                repo_shap_values = {k: v[idx] for k, v in shap_values.items()}
            
            # Generate comparative advantages and improvements
            comparative_advantages = self._generate_comparative_advantages(
                repo_features, features_df, positive_features)
            
            areas_for_improvement = self._generate_improvement_areas(
                repo_features, features_df, negative_features)
            
            # Create explanation object
            explanation = RepositoryExplanation(
                repo_name=repo_name,
                rank=int(row['rank']),
                final_score=float(row['final_score']),
                component_scores={
                    'llm_preference': float(row['llm_preference_score']),
                    'technical_execution': float(row['technical_execution_score']),
                    'market_adoption': float(row['market_adoption_score']),
                    'team_resilience': float(row['team_resilience_score'])
                },
                top_positive_features=positive_features,
                top_negative_features=negative_features,
                shap_values=repo_shap_values,
                human_readable_summary="",  # Will be filled below
                comparative_advantages=comparative_advantages,
                areas_for_improvement=areas_for_improvement
            )
            
            # Generate human-readable summary
            explanation.human_readable_summary = self._generate_human_readable_summary(explanation)
            
            explanations.append(explanation)
        
        self.explanations = explanations
        logger.info(f"Generated explanations for {len(explanations)} repositories")
        
        return explanations
    
    def generate_comparative_analysis(self, repo_a: str, repo_b: str) -> Dict[str, Any]:
        """Generate comparative analysis between two repositories"""
        
        # Find explanations for both repositories
        explanation_a = next((e for e in self.explanations if e.repo_name == repo_a), None)
        explanation_b = next((e for e in self.explanations if e.repo_name == repo_b), None)
        
        if not explanation_a or not explanation_b:
            return {"error": "One or both repositories not found"}
        
        # Compare rankings
        rank_diff = explanation_a.rank - explanation_b.rank
        score_diff = explanation_a.final_score - explanation_b.final_score
        
        # Compare component scores
        component_comparison = {}
        for component in explanation_a.component_scores:
            diff = explanation_a.component_scores[component] - explanation_b.component_scores[component]
            component_comparison[component] = {
                f'{repo_a}_score': explanation_a.component_scores[component],
                f'{repo_b}_score': explanation_b.component_scores[component],
                'difference': diff,
                'winner': repo_a if diff > 0 else repo_b if diff < 0 else 'tie'
            }
        
        # Generate narrative
        if rank_diff < 0:
            narrative = f"{repo_a} ranks higher than {repo_b} (#{explanation_a.rank} vs #{explanation_b.rank}) "
            narrative += f"with a score advantage of {score_diff:.3f}. "
        elif rank_diff > 0:
            narrative = f"{repo_b} ranks higher than {repo_a} (#{explanation_b.rank} vs #{explanation_a.rank}) "
            narrative += f"with a score advantage of {-score_diff:.3f}. "
        else:
            narrative = f"{repo_a} and {repo_b} are tied at rank #{explanation_a.rank}. "
        
        # Add component insights
        strongest_component_a = max(explanation_a.component_scores.items(), key=lambda x: x[1])
        strongest_component_b = max(explanation_b.component_scores.items(), key=lambda x: x[1])
        
        narrative += f"{repo_a}'s strength is {strongest_component_a[0].replace('_', ' ')} "
        narrative += f"({strongest_component_a[1]:.2f}), while {repo_b}'s is "
        narrative += f"{strongest_component_b[0].replace('_', ' ')} ({strongest_component_b[1]:.2f})."
        
        return {
            'repository_a': repo_a,
            'repository_b': repo_b,
            'rank_difference': rank_diff,
            'score_difference': score_diff,
            'component_comparison': component_comparison,
            'narrative_explanation': narrative,
            'winner': repo_a if score_diff > 0 else repo_b if score_diff < 0 else 'tie'
        }
    
    def save_explanations(self, output_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save detailed explanations to JSON file"""
        
        # Convert explanations to serializable format
        explanations_data = []
        
        for explanation in self.explanations:
            explanation_dict = {
                'repo_name': explanation.repo_name,
                'rank': explanation.rank,
                'final_score': explanation.final_score,
                'component_scores': explanation.component_scores,
                'human_readable_summary': explanation.human_readable_summary,
                'comparative_advantages': explanation.comparative_advantages,
                'areas_for_improvement': explanation.areas_for_improvement,
                'top_positive_features': [
                    {
                        'feature_name': f.feature_name,
                        'contribution': f.contribution,
                        'feature_value': f.feature_value,
                        'percentile_rank': f.percentile_rank,
                        'importance_category': f.importance_category
                    }
                    for f in explanation.top_positive_features
                ],
                'top_negative_features': [
                    {
                        'feature_name': f.feature_name,
                        'contribution': f.contribution,
                        'feature_value': f.feature_value,
                        'percentile_rank': f.percentile_rank,
                        'importance_category': f.importance_category
                    }
                    for f in explanation.top_negative_features
                ],
                'shap_values': explanation.shap_values if explanation.shap_values else {}
            }
            explanations_data.append(explanation_dict)
        
        # Create full output
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'total_repositories': len(self.explanations),
            'shap_available': SHAP_AVAILABLE,
            'explanations': explanations_data
        }
        
        if metadata:
            output_data.update(metadata)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Explanations saved to {output_path}")
        
        return output_path

def main():
    """Main execution function for Task 4.2: Explainability and Reasoning"""
    logger.info("Starting BSV Explainability Analysis - Task 4.2")
    
    # Initialize analyzer
    analyzer = ExplainabilityAnalyzer()
    
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
        scores_df, features_df = analyzer.load_scored_data(str(scores_path), str(features_path))
        
        # Generate explanations
        explanations = analyzer.analyze_repository_explanations(scores_df, features_df)
        
        # Save explanations
        output_path = project_root / "data" / "task4_explanations.json"
        analyzer.save_explanations(str(output_path))
        
        # Display summary
        print("\n" + "="*60)
        print("🎉 TASK 4.2 EXPLAINABILITY ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Repositories analyzed: {len(explanations)}")
        print(f"🧠 SHAP values: {'✅ Available' if SHAP_AVAILABLE else '❌ Not available'}")
        print()
        print("📁 Output file:")
        print(f"   • Explanations: {output_path}")
        print()
        
        # Show sample explanations
        print("🔍 Sample Explanations:")
        for i, explanation in enumerate(explanations[:3], 1):
            print(f"\n{i}. {explanation.human_readable_summary}")
            
            if explanation.comparative_advantages:
                print(f"   Advantages: {', '.join(explanation.comparative_advantages)}")
            
            if explanation.top_positive_features:
                top_feature = explanation.top_positive_features[0]
                print(f"   Key strength: {top_feature.feature_name} "
                     f"({top_feature.percentile_rank:.0f}th percentile)")
        
        # Generate sample comparative analysis
        if len(explanations) >= 2:
            print(f"\n🔄 Sample Comparison: {explanations[0].repo_name} vs {explanations[1].repo_name}")
            comparison = analyzer.generate_comparative_analysis(
                explanations[0].repo_name, explanations[1].repo_name)
            print(f"   {comparison['narrative_explanation']}")
        
        return explanations
        
    except Exception as e:
        logger.error(f"Explainability analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
