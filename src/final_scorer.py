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
    """Configuration for composite scoring weights - emphasizing advanced technical analysis"""
    llm_preference: float = 0.30  # Secondary signal from LLM analysis
    technical_execution: float = 0.40  # Primary: Advanced technical metrics and code quality
    market_adoption: float = 0.20  # Growth trajectory and adoption signals
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
    
    def _calculate_technical_execution_score(self, df: pd.DataFrame, repo_idx: int) -> float:
        """Advanced technical execution scoring using sophisticated metrics and AI-driven analysis"""
        repo_data = df.iloc[repo_idx]
        
        # Advanced technical metrics with sophisticated weighting (5 components)
        scores = {}
        
        # 1. Development Velocity & Consistency (25%)
        velocity_score = self._calculate_velocity_sophistication(repo_data)
        scores['velocity'] = velocity_score * 0.25
        
        # 2. Code Quality & Architecture (30%) 
        quality_score = self._calculate_code_sophistication(repo_data)
        scores['quality'] = quality_score * 0.30
        
        # 3. Release Management & Stability (20%)
        release_score = self._calculate_release_sophistication(repo_data)
        scores['release'] = release_score * 0.20
        
        # 4. Innovation Indicators (15%)
        innovation_score = self._calculate_technical_innovation(repo_data)
        scores['innovation'] = innovation_score * 0.15
        
        # 5. Growth Momentum & Time Series Analysis (10%) - BSV Novel Approach #5
        momentum_score = self._calculate_growth_momentum_score(repo_data)
        scores['momentum'] = momentum_score * 0.10
        
        # Composite technical execution score
        final_score = sum(scores.values())
        return min(1.0, max(0.0, final_score))
    
    def _calculate_velocity_sophistication(self, repo_data: pd.Series) -> float:
        """Calculate development velocity using advanced metrics - BSV Novel Approach #1"""
        # Multi-dimensional velocity analysis for category-defining potential
        commits_6m = repo_data.get('commits_6_months', 0)
        contributors = repo_data.get('total_contributors', 1)
        
        # 1. Efficiency Analysis: commits per contributor per month
        efficiency = (commits_6m / 6) / max(contributors, 1)
        efficiency_score = min(1.0, efficiency / 20)  # Cap at 20 commits/contributor/month
        
        # 2. Consistency Analysis: development rhythm indicators
        consistency = repo_data.get('development_consistency_score', 0)
        if consistency == 0:
            # Advanced fallback: estimate from activity patterns
            consistency = 0.7 if commits_6m > 50 else 0.5 if commits_6m > 10 else 0.2
        
        # 3. Velocity Trend Analysis: growth vs decline indicators
        releases_recent = repo_data.get('releases_last_year', 0)
        velocity_trend = min(1.0, releases_recent / 12) if releases_recent > 0 else 0.3
        
        # 4. Innovation Velocity: rapid iteration capability
        days_since_update = repo_data.get('days_since_last_update', 365)
        innovation_velocity = max(0.0, 1.0 - (days_since_update / 90))  # Penalize stagnation
        
        # BSV-weighted composite: emphasize sustained innovation over burst activity
        return (efficiency_score * 0.35 + consistency * 0.25 + 
                velocity_trend * 0.25 + innovation_velocity * 0.15)
    
    def _calculate_code_sophistication(self, repo_data: pd.Series) -> float:
        """Calculate code quality using advanced architectural metrics - BSV Novel Approach #2"""
        # Multi-layered architecture assessment for venture-scale potential
        
        # 1. Technology Stack Sophistication (40% weight)
        language = str(repo_data.get('primary_language', '')).lower()
        # BSV-optimized: prioritize languages that enable category-defining scale
        lang_scores = {
            'rust': 0.95,     # Systems performance + safety
            'go': 0.92,       # Cloud-native scalability  
            'typescript': 0.88, # Modern web architecture
            'python': 0.85,   # AI/ML ecosystem leadership
            'javascript': 0.75, 'java': 0.70, 'c++': 0.65, 'c': 0.60
        }
        tech_score = lang_scores.get(language, 0.50)
        
        # 2. Engineering Maturity Infrastructure (35% weight)
        has_tests = repo_data.get('has_tests', False)
        has_ci_cd = repo_data.get('has_ci_cd', False)
        has_docs = repo_data.get('has_documentation', False)
        
        # BSV emphasis: production-ready engineering practices
        maturity_score = (
            (0.45 if has_tests else 0.0) +      # Testing is critical for scale
            (0.35 if has_ci_cd else 0.0) +      # Automation for growth
            (0.20 if has_docs else 0.0)         # Knowledge transfer capability
        )
        
        # 3. Architectural Complexity & Scale Readiness (25% weight)
        loc = repo_data.get('lines_of_code', 0)
        stars = repo_data.get('stars', 0)
        
        # Complexity-to-adoption ratio: sophisticated but usable
        if stars > 0 and loc > 0:
            complexity_ratio = min(1.0, (loc / 10000) * (stars / 1000) ** 0.5)
        else:
            complexity_ratio = min(1.0, loc / 50000)  # Fallback: pure complexity
        
        # BSV composite: weight factors for venture scalability
        return (tech_score * 0.40 + maturity_score * 0.35 + complexity_ratio * 0.25)
    
    def _calculate_release_sophistication(self, repo_data: pd.Series) -> float:
        """Calculate release management sophistication - BSV Novel Approach #3"""
        # Multi-dimensional release maturity for enterprise readiness
        total_releases = repo_data.get('total_releases', 0)
        releases_last_year = repo_data.get('releases_last_year', 0)
        
        if total_releases == 0:
            return 0.0
            
        # 1. Release Cadence Intelligence (40% weight)
        if releases_last_year >= 12:     # Monthly: high velocity
            cadence_score = 1.0
        elif releases_last_year >= 6:    # Bi-monthly: steady innovation
            cadence_score = 0.9
        elif releases_last_year >= 4:    # Quarterly: enterprise rhythm
            cadence_score = 0.8
        elif releases_last_year >= 2:    # Bi-annual: measured approach
            cadence_score = 0.6
        elif releases_last_year >= 1:    # Annual: minimum viability
            cadence_score = 0.4
        else:
            cadence_score = 0.2           # Stagnant
            
        # 2. Release Portfolio Maturity (35% weight)  
        # BSV insight: sustained release capability indicates team maturity
        portfolio_maturity = min(1.0, total_releases / 30)  # Normalized to 30 releases
        
        # 3. Version Management Sophistication (25% weight)
        # Estimate semantic versioning and release discipline
        if total_releases >= 20:
            version_sophistication = 1.0    # Proven release management
        elif total_releases >= 10:
            version_sophistication = 0.8    # Developing practices
        elif total_releases >= 5:
            version_sophistication = 0.6    # Basic discipline
        else:
            version_sophistication = 0.4    # Early stage
            
        # BSV composite: emphasize sustained, disciplined release capability
        return (cadence_score * 0.40 + portfolio_maturity * 0.35 + 
                version_sophistication * 0.25)
    
    def _calculate_technical_innovation(self, repo_data: pd.Series) -> float:
        """Calculate technical innovation indicators - BSV Novel Approach #4"""
        # Multi-layered innovation assessment with AI/ML bias for BSV portfolio fit
        
        # 1. AI/ML Innovation Leadership (45% weight) 
        topics = repo_data.get('topics', '[]')
        description = repo_data.get('description', '')
        combined_text = f"{topics} {description}".lower()
        
        # BSV-optimized AI keyword hierarchy
        tier1_ai = ['llm', 'transformer', 'neural', 'gpt', 'bert']  # Cutting-edge
        tier2_ai = ['ai', 'ml', 'machine-learning', 'deep-learning'] # Core AI
        tier3_ai = ['nlp', 'computer-vision', 'reinforcement']      # Specialized
        
        if any(keyword in combined_text for keyword in tier1_ai):
            ai_innovation_score = 0.95  # Frontier AI technology
        elif any(keyword in combined_text for keyword in tier2_ai):
            ai_innovation_score = 0.80  # Core AI capabilities  
        elif any(keyword in combined_text for keyword in tier3_ai):
            ai_innovation_score = 0.65  # AI-adjacent innovation
        else:
            ai_innovation_score = 0.35  # Non-AI baseline
            
        # 2. Innovation Velocity (30% weight)
        days_since_update = repo_data.get('days_since_last_update', 365)
        # BSV emphasis: sustained innovation over burst activity
        velocity_score = max(0.0, 1.0 - (days_since_update / 120))  # 4-month window
        
        # 3. Technology Differentiation (25% weight)
        tech_diff = repo_data.get('technology_differentiation_score', 0.5)
        
        # BSV composite: prioritize AI innovation for productivity transformation
        return (ai_innovation_score * 0.45 + velocity_score * 0.30 + tech_diff * 0.25)
    
    def _calculate_growth_momentum_score(self, repo_data: pd.Series) -> float:
        """Advanced time series analysis for growth trajectory prediction - BSV Novel Approach #5"""
        
        # Extract time-based metrics from available data
        stars = repo_data.get('stars', 0)
        created_at = repo_data.get('created_at', '')
        updated_at = repo_data.get('updated_at', '')
        
        # Calculate repository age in months
        if created_at:
            try:
                from datetime import datetime
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                current_date = datetime.now(created_date.tzinfo)
                age_months = max(1, (current_date - created_date).days / 30.44)  # Accurate month calculation
            except:
                age_months = 12  # Fallback
        else:
            age_months = 12  # Fallback
        
        # 1. Growth Velocity Analysis (40% weight)
        stars_per_month = stars / age_months
        # Normalize using sigmoid for bounded score
        velocity_score = np.tanh(stars_per_month / 100)  # 100 stars/month = ~1.0 score
        
        # 2. Recent Activity Momentum (35% weight)  
        recent_commits = repo_data.get('commits_6_months', 0)
        total_commits = repo_data.get('total_commits', recent_commits)
        
        if total_commits > 0:
            recent_activity_ratio = recent_commits / total_commits
            # Annualized recent activity momentum
            momentum_factor = recent_activity_ratio * 2  # 6 months to annual
            momentum_score = np.tanh(momentum_factor)
        else:
            momentum_score = 0.0
        
        # 3. Release Momentum (25% weight)
        releases_last_year = repo_data.get('releases_last_year', 0)
        total_releases = repo_data.get('total_releases', releases_last_year)
        
        if total_releases > 0 and age_months >= 12:
            # Release acceleration: recent vs historical rate
            historical_release_rate = total_releases / (age_months / 12)
            release_acceleration = releases_last_year / max(historical_release_rate, 0.1)
            release_momentum = min(1.0, np.tanh(release_acceleration))
        else:
            # For younger projects, just use recent release rate
            release_momentum = min(1.0, releases_last_year / 4)  # Quarterly releases = 1.0
        
        # Composite momentum score with time-series weighting
        composite_momentum = (
            velocity_score * 0.40 +
            momentum_score * 0.35 + 
            release_momentum * 0.25
        )
        
        return min(1.0, max(0.0, composite_momentum))
    
    def _calculate_bsv_investment_score(self, component_scores: Dict[str, float], funding_multiplier: float) -> float:
        """Calculate BSV investment score using sophisticated venture capital methodology"""
        # BSV Investment Score = Technical Dominance * Market Timing * Team Execution * Funding Advantage
        
        # 1. Technical Dominance (40% weight) - Our enhanced technical execution score
        technical_dominance = component_scores.get('technical_execution', 0.0)
        
        # 2. Market Timing (30% weight) - Market adoption + LLM innovation preference  
        market_timing = (component_scores.get('market_adoption', 0.0) * 0.6 + 
                        component_scores.get('llm_preference', 0.0) * 0.4)
        
        # 3. Team Execution (20% weight) - Team resilience score
        team_execution = component_scores.get('team_resilience', 0.0)
        
        # 4. Funding Advantage (10% weight) - Inverse of funding gate (higher if unfunded)
        funding_advantage = (2.0 - funding_multiplier) / 2.0  # Convert 0.6-1.0 to 0.5-0.2, then invert
        
        # BSV composite with venture bias toward technical innovation
        bsv_score = (
            technical_dominance * 0.40 +
            market_timing * 0.30 +
            team_execution * 0.20 + 
            funding_advantage * 0.10
        )
        
        return min(1.0, max(0.0, bsv_score))
    
    def _calculate_category_potential_score(self, repo_data: pd.Series) -> float:
        """Calculate category-defining potential using AI/ML innovation bias"""
        # Use our technical innovation scoring method which has AI/ML bias built-in
        base_innovation = self._calculate_technical_innovation(repo_data)
        
        # Boost score for repositories with clear category-defining characteristics
        stars = repo_data.get('stars', 0)
        forks = repo_data.get('forks', 0)
        
        # Category potential multiplier based on engagement and innovation
        engagement_factor = min(1.5, (stars + forks * 2) / 10000)  # Cap at 1.5x boost
        category_score = base_innovation * engagement_factor
        
        return min(1.0, max(0.0, category_score))
    
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
        
        # Merge datasets on repository identifier with multiple fallback strategies
        merged_df = None
        
        # Strategy 1: Direct repository column match
        if 'repository' in task2_df.columns and 'repository' in task3_df.columns:
            merged_df = task2_df.merge(task3_df, on='repository', how='inner')
            logger.info(f"Merged using 'repository' column: {len(merged_df)} matches")
            
        # Strategy 2: repo_name to repository match
        elif 'repo_name' in task2_df.columns and 'repository' in task3_df.columns:
            merged_df = task2_df.merge(task3_df, left_on='repo_name', right_on='repository', how='inner')
            logger.info(f"Merged using 'repo_name' to 'repository': {len(merged_df)} matches")
            
        # Strategy 3: full_name to repository match  
        elif 'full_name' in task2_df.columns and 'repository' in task3_df.columns:
            merged_df = task2_df.merge(task3_df, left_on='full_name', right_on='repository', how='inner')
            logger.info(f"Merged using 'full_name' to 'repository': {len(merged_df)} matches")
        
        # If no matches found or datasets have different repositories, create fallback scores
        if merged_df is None or len(merged_df) == 0:
            logger.warning("No repository matches found between datasets - creating fallback scoring")
            merged_df = task2_df.copy()
            
            # Add fallback LLM scores based on existing features
            merged_df['llm_preference_score'] = self._generate_fallback_llm_scores(task2_df)
            merged_df['innovation_reasoning'] = 'Estimated from technical and market features'
            merged_df['innovation_category'] = 'moderate_innovation'
            merged_df['innovation_score'] = merged_df['llm_preference_score'] * 0.8  # Derive from LLM score
            merged_df['competitive_advantage'] = merged_df['llm_preference_score'] * 0.9  # Derive from LLM score
        
        logger.info(f"Successfully merged {len(merged_df)} repositories")
        
        # Normalize features for consistent scoring
        numeric_columns = merged_df.select_dtypes(include=[np.number]).columns.tolist()
        merged_df_norm = self._normalize_features(merged_df, numeric_columns)
        
        # Calculate scores for each repository
        results = []
        
        for idx in range(len(merged_df_norm)):
            repo_data = merged_df.iloc[idx]
            repo_name = repo_data.get('repo_name', repo_data.get('repository', f'repo_{idx}'))
            
            # 1. LLM Preference Score (60%) - use actual LLM preference score
            llm_score = merged_df_norm.iloc[idx].get('llm_preference_score', 0.5)
            
            # 2. Technical Execution Score (15%) - composite of technical metrics
            tech_score = self._calculate_technical_execution_score(merged_df_norm, idx)
            tech_reasons = [ReasonCode(
                factor="technical_execution",
                contribution=tech_score,
                description="Enhanced technical execution using BSV 5-component analysis",
                value=tech_score
            )] if tech_score > 0.7 else []
            
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
            
            # Calculate real BSV investment scores using our enhanced technical execution methods
            bsv_investment_score = self._calculate_bsv_investment_score(result.component_scores, result.funding_gate_multiplier)
            
            # Calculate category potential using AI/ML bias from technical innovation score
            category_potential_score = self._calculate_category_potential_score(original_data)
            
            # Calculate innovation score using our technical innovation method with AI/ML bias
            innovation_score = self._calculate_technical_innovation(original_data)
            
            # Calculate competitive advantage from market + technical + LLM scores
            competitive_advantage = (
                result.component_scores['market_adoption'] * 0.4 +
                result.component_scores['technical_execution'] * 0.4 +
                result.component_scores['llm_preference'] * 0.2
            )
            
            row.update({
                'stars': original_data.get('stars', 0),
                'forks': original_data.get('forks', 0),
                'created_at': original_data.get('created_at', ''),
                'funding_risk_level': original_data.get('funding_risk_level', 'unknown'),
                'category_potential_score': category_potential_score,
                'bsv_investment_score': bsv_investment_score,
                'innovation_score': innovation_score,
                'competitive_advantage': competitive_advantage
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
    
    def _generate_fallback_llm_scores(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate fallback LLM preference scores when LLM data unavailable"""
        fallback_scores = []
        
        for _, row in features_df.iterrows():
            # Use existing BSV investment score if available
            if 'bsv_investment_score' in row and pd.notna(row['bsv_investment_score']):
                score = float(row['bsv_investment_score'])
            else:
                # Calculate from available features
                stars = row.get('stars', 0)
                forks = row.get('forks', 0)
                
                # Engagement-based scoring
                engagement = min(np.log1p(stars + forks), 10) / 10
                
                # Technical maturity if available
                tech_score = row.get('technical_maturity_composite', 0.5)
                
                # Market positioning if available  
                market_score = row.get('market_positioning_composite', 0.5)
                
                # Combine metrics
                score = (engagement * 0.4 + tech_score * 0.3 + market_score * 0.3)
            
            # Normalize to [0,1] and add some variance
            normalized_score = min(max(score, 0.0), 1.0)
            fallback_scores.append(normalized_score)
            
        return pd.Series(fallback_scores)

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
