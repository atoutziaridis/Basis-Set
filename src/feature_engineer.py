"""
Feature Engineering Pipeline for BSV Repository Prioritizer
Transforms raw GitHub data into meaningful signals for investment analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
import re

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for GitHub repository analysis
    Transforms raw data into investment-grade signals
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_metadata = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load enriched data from Task 1"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data: {len(df)} repositories, {len(df.columns)} raw features")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def engineer_execution_velocity_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtask 2.1: Engineer execution and velocity signals
        Focus on development pace, consistency, and maintenance activity
        """
        logger.info("Engineering execution & velocity signals...")
        
        # Robust commit velocity (Theil-Sen slope estimation)
        df['commit_velocity_score'] = self._calculate_commit_velocity(df)
        
        # Release cadence quality
        df['release_cadence_score'] = self._calculate_release_cadence(df)
        
        # Maintenance activity index
        df['maintenance_activity_score'] = self._calculate_maintenance_activity(df)
        
        # Development consistency (coefficient of variation)
        df['development_consistency_score'] = self._calculate_development_consistency(df)
        
        # Composite execution score
        df['execution_velocity_composite'] = self._calculate_execution_composite(df)
        
        logger.info("✅ Execution & velocity signals engineered")
        return df
    
    def _calculate_commit_velocity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate robust commit velocity score"""
        # Use Theil-Sen slope estimation for robustness
        velocity_scores = []
        
        for _, row in df.iterrows():
            try:
                # Normalize by repository age and size
                commits_6m = row.get('commits_6_months', 0)
                active_weeks = row.get('active_weeks_6_months', 1)
                repo_age_months = self._calculate_repo_age_months(row.get('created_at'))
                
                # Calculate normalized velocity
                if active_weeks > 0 and repo_age_months > 0:
                    # Commits per active week, adjusted for repo maturity
                    base_velocity = commits_6m / active_weeks
                    # Bonus for sustained activity over time
                    consistency_bonus = min(active_weeks / 26, 1.0)  # 26 weeks = 6 months
                    # Age adjustment (newer repos get slight bonus, older repos need consistency)
                    age_factor = 1.0 + (0.2 / (1 + repo_age_months / 12))
                    
                    velocity_score = base_velocity * consistency_bonus * age_factor
                else:
                    velocity_score = 0
                    
                velocity_scores.append(velocity_score)
                
            except Exception as e:
                logger.warning(f"Error calculating commit velocity: {e}")
                velocity_scores.append(0)
        
        # Normalize to 0-1 scale using robust scaling
        velocity_array = np.array(velocity_scores)
        if velocity_array.max() > 0:
            # Use 95th percentile as max to avoid outlier impact
            max_val = np.percentile(velocity_array[velocity_array > 0], 95)
            normalized = np.clip(velocity_array / max_val, 0, 1)
            return pd.Series(normalized)
        else:
            return pd.Series([0] * len(velocity_scores))
    
    def _calculate_release_cadence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate release cadence quality score"""
        cadence_scores = []
        
        for _, row in df.iterrows():
            try:
                total_releases = row.get('total_releases', 0)
                releases_last_year = row.get('releases_last_year', 0)
                days_since_last = row.get('days_since_last_release', float('inf'))
                
                if total_releases == 0:
                    cadence_scores.append(0)
                    continue
                
                # Base score from release activity
                base_score = min(releases_last_year / 4, 1.0)  # 4 releases/year = max score
                
                # Penalty for long gaps
                recency_factor = 1.0
                if days_since_last < float('inf'):
                    if days_since_last > 365:  # Over 1 year
                        recency_factor = 0.3
                    elif days_since_last > 180:  # Over 6 months
                        recency_factor = 0.7
                    elif days_since_last > 90:   # Over 3 months
                        recency_factor = 0.9
                
                # Consistency bonus for regular releases
                consistency_bonus = 1.0
                if total_releases >= 3:
                    consistency_bonus = 1.2
                
                final_score = base_score * recency_factor * consistency_bonus
                cadence_scores.append(min(final_score, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating release cadence: {e}")
                cadence_scores.append(0)
        
        return pd.Series(cadence_scores)
    
    def _calculate_maintenance_activity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate maintenance activity score"""
        maintenance_scores = []
        
        for _, row in df.iterrows():
            try:
                # Recent activity indicators
                commits_30_days = row.get('commits_30_days', 0)
                issues_30_days = row.get('issues_30_days', 0)
                prs_30_days = row.get('prs_30_days', 0)
                
                # Response time quality
                response_time_hours = row.get('median_issue_response_time_hours')
                
                # Activity score (normalized)
                activity_score = min((commits_30_days + issues_30_days + prs_30_days) / 20, 1.0)
                
                # Responsiveness score
                responsiveness_score = 1.0
                if response_time_hours is not None and response_time_hours > 0:
                    if response_time_hours <= 24:      # Within 1 day
                        responsiveness_score = 1.0
                    elif response_time_hours <= 168:   # Within 1 week
                        responsiveness_score = 0.8
                    elif response_time_hours <= 720:   # Within 1 month
                        responsiveness_score = 0.6
                    else:
                        responsiveness_score = 0.3
                
                # Combined maintenance score
                maintenance_score = (activity_score * 0.6) + (responsiveness_score * 0.4)
                maintenance_scores.append(maintenance_score)
                
            except Exception as e:
                logger.warning(f"Error calculating maintenance activity: {e}")
                maintenance_scores.append(0)
        
        return pd.Series(maintenance_scores)
    
    def _calculate_development_consistency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate development consistency (inverse of coefficient of variation)"""
        consistency_scores = []
        
        for _, row in df.iterrows():
            try:
                commits_6m = row.get('commits_6_months', 0)
                active_weeks = row.get('active_weeks_6_months', 1)
                avg_commits_per_week = row.get('avg_commits_per_week', 0)
                
                if active_weeks <= 1 or avg_commits_per_week <= 0:
                    consistency_scores.append(0)
                    continue
                
                # Estimate coefficient of variation (lower = more consistent)
                # If commits are spread evenly across active weeks, CV should be low
                expected_variance = max(avg_commits_per_week * 0.5, 1)  # Assume some natural variation
                cv_estimate = expected_variance / avg_commits_per_week
                
                # Convert to consistency score (1 - normalized CV)
                consistency_score = max(1 - (cv_estimate / 2), 0)  # CV of 2 = 0 consistency
                
                # Bonus for sustained activity
                sustainability_bonus = min(active_weeks / 20, 1.0)  # 20 weeks = good sustainability
                
                final_consistency = consistency_score * sustainability_bonus
                consistency_scores.append(final_consistency)
                
            except Exception as e:
                logger.warning(f"Error calculating development consistency: {e}")
                consistency_scores.append(0)
        
        return pd.Series(consistency_scores)
    
    def _calculate_execution_composite(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite execution score"""
        # Weighted combination of execution signals
        weights = {
            'commit_velocity_score': 0.35,
            'release_cadence_score': 0.25,
            'maintenance_activity_score': 0.25,
            'development_consistency_score': 0.15
        }
        
        composite_scores = []
        for _, row in df.iterrows():
            weighted_sum = sum(row.get(signal, 0) * weight 
                             for signal, weight in weights.items())
            composite_scores.append(weighted_sum)
        
        return pd.Series(composite_scores)
    
    def engineer_team_community_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtask 2.2: Engineer team and community signals
        Focus on team resilience, community health, and growth patterns
        """
        logger.info("Engineering team & community signals...")
        
        # Enhanced bus factor analysis
        df['team_resilience_score'] = self._calculate_team_resilience(df)
        
        # Community health indicators
        df['community_health_score'] = self._calculate_community_health(df)
        
        # Growth trajectory analysis
        df['growth_trajectory_score'] = self._calculate_growth_trajectory(df)
        
        # Network effects scoring
        df['network_effects_score'] = self._calculate_network_effects(df)
        
        # Composite team & community score
        df['team_community_composite'] = self._calculate_team_community_composite(df)
        
        logger.info("✅ Team & community signals engineered")
        return df
    
    def _calculate_team_resilience(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced team resilience beyond basic bus factor"""
        resilience_scores = []
        
        for _, row in df.iterrows():
            try:
                bus_factor = row.get('bus_factor', 0)
                total_contributors = row.get('total_contributors', 1)
                active_contributors = row.get('active_contributors', 1)
                contribution_gini = row.get('contribution_gini', 1)
                
                # Base resilience from bus factor (higher = better)
                base_resilience = bus_factor
                
                # Team size factor (more contributors = more resilient, with diminishing returns)
                team_size_factor = min(np.log(total_contributors + 1) / np.log(10), 1.0)
                
                # Active contributor ratio
                activity_ratio = active_contributors / max(total_contributors, 1)
                
                # Contribution equality (lower Gini = more equal = more resilient)
                equality_score = 1 - contribution_gini
                
                # Combined resilience
                resilience_score = (base_resilience * 0.4 + 
                                  team_size_factor * 0.3 + 
                                  activity_ratio * 0.2 + 
                                  equality_score * 0.1)
                
                resilience_scores.append(min(resilience_score, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating team resilience: {e}")
                resilience_scores.append(0)
        
        return pd.Series(resilience_scores)
    
    def _calculate_community_health(self, df: pd.DataFrame) -> pd.Series:
        """Calculate community health score"""
        health_scores = []
        
        for _, row in df.iterrows():
            try:
                stars = row.get('stars', 0)
                forks = row.get('forks', 0)
                watchers = row.get('subscribers_count', 0)
                fork_to_star_ratio = row.get('fork_to_star_ratio', 0)
                
                # Engagement quality (fork ratio indicates developer interest)
                engagement_quality = min(fork_to_star_ratio * 10, 1.0)  # 0.1 ratio = max score
                
                # Community size (log scale to handle wide ranges)
                community_size_score = min(np.log(stars + 1) / np.log(1000), 1.0)  # 1000 stars = max score
                
                # Active watching vs passive starring
                if stars > 0:
                    watch_ratio = watchers / stars
                    attention_quality = min(watch_ratio * 5, 1.0)  # 0.2 ratio = max score
                else:
                    attention_quality = 0
                
                # Combined community health
                health_score = (engagement_quality * 0.4 + 
                              community_size_score * 0.4 + 
                              attention_quality * 0.2)
                
                health_scores.append(health_score)
                
            except Exception as e:
                logger.warning(f"Error calculating community health: {e}")
                health_scores.append(0)
        
        return pd.Series(health_scores)
    
    def _calculate_growth_trajectory(self, df: pd.DataFrame) -> pd.Series:
        """Calculate growth trajectory score"""
        trajectory_scores = []
        
        for _, row in df.iterrows():
            try:
                stars_per_month = row.get('stars_per_month', 0)
                repo_age_months = self._calculate_repo_age_months(row.get('created_at'))
                total_stars = row.get('stars', 0)
                
                if repo_age_months <= 0:
                    trajectory_scores.append(0)
                    continue
                
                # Growth rate score (normalized)
                growth_rate_score = min(stars_per_month / 50, 1.0)  # 50 stars/month = max score
                
                # Acceleration bonus for newer repos with high growth
                if repo_age_months <= 12 and stars_per_month > 20:  # New repo, high growth
                    acceleration_bonus = 1.2
                elif repo_age_months <= 24 and stars_per_month > 10:  # Young repo, good growth
                    acceleration_bonus = 1.1
                else:
                    acceleration_bonus = 1.0
                
                # Sustainability check (very high growth on tiny repos might not be sustainable)
                if total_stars < 100 and stars_per_month > 100:
                    sustainability_factor = 0.8  # Slight discount for potentially unsustainable growth
                else:
                    sustainability_factor = 1.0
                
                trajectory_score = growth_rate_score * acceleration_bonus * sustainability_factor
                trajectory_scores.append(min(trajectory_score, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating growth trajectory: {e}")
                trajectory_scores.append(0)
        
        return pd.Series(trajectory_scores)
    
    def _calculate_network_effects(self, df: pd.DataFrame) -> pd.Series:
        """Calculate network effects score"""
        network_scores = []
        
        for _, row in df.iterrows():
            try:
                dependents_count = row.get('dependents_count', 0)
                network_count = row.get('network_count', 0)
                has_package = row.get('has_package', False)
                
                # Package downloads (any platform)
                pypi_downloads = row.get('pypi_downloads', 0)
                npm_downloads = row.get('npm_downloads', 0)
                cargo_downloads = row.get('cargo_downloads', 0)
                total_downloads = pypi_downloads + npm_downloads + cargo_downloads
                
                # Dependents score
                dependents_score = min(np.log(dependents_count + 1) / np.log(100), 1.0)  # 100 dependents = max score
                
                # Package adoption score
                if has_package and total_downloads > 0:
                    package_score = min(np.log(total_downloads + 1) / np.log(10000), 1.0)  # 10k downloads = max score
                else:
                    package_score = 0
                
                # Network reach score
                network_reach_score = min(np.log(network_count + 1) / np.log(1000), 1.0)  # 1000 network = max score
                
                # Combined network effects
                network_score = (dependents_score * 0.5 + 
                               package_score * 0.3 + 
                               network_reach_score * 0.2)
                
                network_scores.append(network_score)
                
            except Exception as e:
                logger.warning(f"Error calculating network effects: {e}")
                network_scores.append(0)
        
        return pd.Series(network_scores)
    
    def _calculate_team_community_composite(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite team & community score"""
        weights = {
            'team_resilience_score': 0.3,
            'community_health_score': 0.3,
            'growth_trajectory_score': 0.25,
            'network_effects_score': 0.15
        }
        
        composite_scores = []
        for _, row in df.iterrows():
            weighted_sum = sum(row.get(signal, 0) * weight 
                             for signal, weight in weights.items())
            composite_scores.append(weighted_sum)
        
        return pd.Series(composite_scores)
    
    def engineer_technical_maturity_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtask 2.3: Engineer technical maturity indicators
        Focus on operational readiness, code quality, and API stability
        """
        logger.info("Engineering technical maturity signals...")
        
        # Operational readiness score
        df['operational_readiness_score'] = self._calculate_operational_readiness(df)
        
        # Code quality indicators
        df['code_quality_score'] = self._calculate_code_quality(df)
        
        # API stability and maturity
        df['api_stability_score'] = self._calculate_api_stability(df)
        
        # Documentation completeness
        df['documentation_score'] = self._calculate_documentation_score(df)
        
        # Composite technical maturity
        df['technical_maturity_composite'] = self._calculate_technical_maturity_composite(df)
        
        logger.info("✅ Technical maturity signals engineered")
        return df
    
    def _calculate_operational_readiness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate operational readiness score"""
        readiness_scores = []
        
        for _, row in df.iterrows():
            try:
                # CI/CD and automation
                has_ci_cd = row.get('has_ci_cd', False)
                workflow_count = row.get('workflow_count', 0)
                
                # Testing infrastructure
                has_tests = row.get('has_tests', False)
                
                # Containerization and deployment
                has_dockerfile = row.get('has_dockerfile', False)
                has_docker_compose = row.get('has_docker_compose', False)
                
                # Build automation
                has_makefile = row.get('has_makefile', False)
                
                # Configuration management
                config_score = row.get('config_completeness_score', 0)
                
                # Scoring
                ci_cd_score = 0.3 if has_ci_cd else 0
                if workflow_count > 1:
                    ci_cd_score += 0.1  # Bonus for multiple workflows
                
                testing_score = 0.25 if has_tests else 0
                
                deployment_score = 0
                if has_dockerfile:
                    deployment_score += 0.15
                if has_docker_compose:
                    deployment_score += 0.1
                
                automation_score = 0.1 if has_makefile else 0
                
                config_score_weighted = config_score * 0.1
                
                total_readiness = (ci_cd_score + testing_score + deployment_score + 
                                 automation_score + config_score_weighted)
                
                readiness_scores.append(min(total_readiness, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating operational readiness: {e}")
                readiness_scores.append(0)
        
        return pd.Series(readiness_scores)
    
    def _calculate_code_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate code quality score"""
        quality_scores = []
        
        for _, row in df.iterrows():
            try:
                # Language diversity (not too fragmented, but not monolithic)
                language_diversity = row.get('language_diversity', 1)
                diversity_score = min(language_diversity / 5, 1.0) * (1.2 - min(language_diversity / 10, 0.2))
                
                # Code organization and linting
                has_eslintrc = row.get('has_eslintrc', False)
                has_prettier = row.get('has_prettier', False)
                has_gitignore = row.get('has_gitignore', False)
                
                # Quality tooling score
                tooling_score = 0
                if has_eslintrc:
                    tooling_score += 0.3
                if has_prettier:
                    tooling_score += 0.2
                if has_gitignore:
                    tooling_score += 0.1
                
                # Repository size factor (very small or very large repos might have quality issues)
                repo_size = row.get('size', 0)
                if 100 <= repo_size <= 50000:  # Reasonable size range
                    size_factor = 1.0
                elif repo_size < 100:  # Very small
                    size_factor = 0.8
                else:  # Very large
                    size_factor = 0.9
                
                combined_quality = (diversity_score * 0.4 + tooling_score * 0.4 + 0.2) * size_factor
                quality_scores.append(min(combined_quality, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating code quality: {e}")
                quality_scores.append(0)
        
        return pd.Series(quality_scores)
    
    def _calculate_api_stability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate API stability score"""
        stability_scores = []
        
        for _, row in df.iterrows():
            try:
                total_releases = row.get('total_releases', 0)
                has_package = row.get('has_package', False)
                
                # Release maturity
                if total_releases == 0:
                    release_maturity = 0
                elif total_releases >= 10:
                    release_maturity = 1.0
                else:
                    release_maturity = total_releases / 10
                
                # Package availability bonus
                package_bonus = 0.2 if has_package else 0
                
                # License clarity (permissive licenses are better for adoption)
                license_name = row.get('license', '')
                if any(permissive in str(license_name).lower() 
                      for permissive in ['mit', 'apache', 'bsd']):
                    license_score = 0.2
                else:
                    license_score = 0.1
                
                stability_score = release_maturity * 0.6 + package_bonus + license_score
                stability_scores.append(min(stability_score, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating API stability: {e}")
                stability_scores.append(0)
        
        return pd.Series(stability_scores)
    
    def _calculate_documentation_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate documentation completeness score"""
        doc_scores = []
        
        for _, row in df.iterrows():
            try:
                readme_quality = row.get('readme_quality_score', 0)
                has_docs_directory = row.get('has_docs_directory', False)
                
                # Base documentation from README quality
                base_doc_score = readme_quality * 0.7
                
                # Bonus for dedicated docs
                docs_bonus = 0.3 if has_docs_directory else 0
                
                total_doc_score = base_doc_score + docs_bonus
                doc_scores.append(min(total_doc_score, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating documentation score: {e}")
                doc_scores.append(0)
        
        return pd.Series(doc_scores)
    
    def _calculate_technical_maturity_composite(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite technical maturity score"""
        weights = {
            'operational_readiness_score': 0.35,
            'code_quality_score': 0.25,
            'api_stability_score': 0.25,
            'documentation_score': 0.15
        }
        
        composite_scores = []
        for _, row in df.iterrows():
            weighted_sum = sum(row.get(signal, 0) * weight 
                             for signal, weight in weights.items())
            composite_scores.append(weighted_sum)
        
        return pd.Series(composite_scores)
    
    def engineer_market_positioning_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtask 2.4: Engineer market positioning signals
        Focus on problem ambition, differentiation, and commercial viability
        """
        logger.info("Engineering market positioning signals...")
        
        # Problem ambition scoring
        df['problem_ambition_score'] = self._calculate_problem_ambition(df)
        
        # Commercial viability indicators
        df['commercial_viability_score'] = self._calculate_commercial_viability(df)
        
        # Technology differentiation
        df['technology_differentiation_score'] = self._calculate_technology_differentiation(df)
        
        # Market readiness indicators
        df['market_readiness_score'] = self._calculate_market_readiness(df)
        
        # Composite market positioning
        df['market_positioning_composite'] = self._calculate_market_positioning_composite(df)
        
        logger.info("✅ Market positioning signals engineered")
        return df
    
    def _calculate_problem_ambition(self, df: pd.DataFrame) -> pd.Series:
        """Calculate problem ambition based on description and topics"""
        ambition_scores = []
        
        # High-ambition keywords/topics
        high_ambition_terms = {
            'enterprise': 0.8, 'scale': 0.7, 'platform': 0.6, 'infrastructure': 0.7,
            'ai': 0.9, 'machine-learning': 0.8, 'automation': 0.7, 'analytics': 0.6,
            'database': 0.6, 'distributed': 0.7, 'cloud': 0.6, 'api': 0.5,
            'framework': 0.6, 'system': 0.6, 'engine': 0.7, 'pipeline': 0.6
        }
        
        # Medium-ambition terms
        medium_ambition_terms = {
            'tool': 0.4, 'library': 0.3, 'utility': 0.3, 'helper': 0.2,
            'client': 0.3, 'wrapper': 0.2, 'gui': 0.3, 'web': 0.4
        }
        
        for _, row in df.iterrows():
            try:
                description = str(row.get('description', '')).lower()
                topics = str(row.get('topics', '')).lower()
                combined_text = f"{description} {topics}"
                
                # Calculate ambition score
                ambition_score = 0
                word_count = 0
                
                # Check for high ambition terms
                for term, score in high_ambition_terms.items():
                    if term in combined_text:
                        ambition_score += score
                        word_count += 1
                
                # Check for medium ambition terms (lower weight)
                for term, score in medium_ambition_terms.items():
                    if term in combined_text:
                        ambition_score += score * 0.5
                        word_count += 0.5
                
                # Normalize by word coverage and cap at 1.0
                if word_count > 0:
                    normalized_score = min(ambition_score / max(word_count * 0.6, 1), 1.0)
                else:
                    normalized_score = 0.2  # Default for projects without clear category
                
                ambition_scores.append(normalized_score)
                
            except Exception as e:
                logger.warning(f"Error calculating problem ambition: {e}")
                ambition_scores.append(0.2)
        
        return pd.Series(ambition_scores)
    
    def _calculate_commercial_viability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate commercial viability score"""
        viability_scores = []
        
        for _, row in df.iterrows():
            try:
                # License permissiveness
                license_name = str(row.get('license', '')).lower()
                if any(permissive in license_name for permissive in ['mit', 'apache', 'bsd']):
                    license_score = 0.3
                elif 'gpl' in license_name:
                    license_score = 0.1  # Less commercially friendly
                else:
                    license_score = 0.2  # Unknown/other
                
                # Package availability (indicates intent to distribute)
                has_package = row.get('has_package', False)
                package_score = 0.2 if has_package else 0
                
                # Documentation quality (important for adoption)
                readme_quality = row.get('readme_quality_score', 0)
                has_docs = row.get('has_docs_directory', False)
                doc_score = (readme_quality * 0.15) + (0.1 if has_docs else 0)
                
                # Deployment readiness
                has_dockerfile = row.get('has_dockerfile', False)
                deployment_score = 0.15 if has_dockerfile else 0
                
                # API/integration indicators
                description = str(row.get('description', '')).lower()
                api_indicators = ['api', 'sdk', 'integration', 'plugin', 'extension']
                api_score = 0.1 if any(term in description for term in api_indicators) else 0
                
                total_viability = license_score + package_score + doc_score + deployment_score + api_score
                viability_scores.append(min(total_viability, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating commercial viability: {e}")
                viability_scores.append(0)
        
        return pd.Series(viability_scores)
    
    def _calculate_technology_differentiation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate technology differentiation score"""
        differentiation_scores = []
        
        for _, row in df.iterrows():
            try:
                # Language choice innovation
                language = str(row.get('primary_language', '')).lower()
                language_diversity = row.get('language_diversity', 1)
                
                # Modern/innovative language bonus
                modern_languages = ['rust', 'go', 'typescript', 'kotlin', 'swift']
                language_score = 0.2 if language in modern_languages else 0.1
                
                # Multi-language sophistication
                diversity_score = min(language_diversity / 5, 0.2)
                
                # Technical sophistication indicators
                description = str(row.get('description', '')).lower()
                topics = str(row.get('topics', '')).lower()
                combined_text = f"{description} {topics}"
                
                sophistication_terms = {
                    'machine learning': 0.3, 'neural network': 0.3, 'deep learning': 0.3,
                    'blockchain': 0.2, 'distributed': 0.2, 'microservices': 0.2,
                    'real-time': 0.2, 'streaming': 0.2, 'concurrent': 0.1,
                    'compiler': 0.3, 'interpreter': 0.2, 'virtual machine': 0.3
                }
                
                sophistication_score = 0
                for term, score in sophistication_terms.items():
                    if term in combined_text:
                        sophistication_score += score
                
                sophistication_score = min(sophistication_score, 0.4)
                
                # Innovation indicators (novel approaches)
                innovation_terms = ['novel', 'new approach', 'innovative', 'breakthrough', 'cutting-edge']
                innovation_score = 0.1 if any(term in combined_text for term in innovation_terms) else 0
                
                total_differentiation = language_score + diversity_score + sophistication_score + innovation_score
                differentiation_scores.append(min(total_differentiation, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating technology differentiation: {e}")
                differentiation_scores.append(0)
        
        return pd.Series(differentiation_scores)
    
    def _calculate_market_readiness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market readiness score"""
        readiness_scores = []
        
        for _, row in df.iterrows():
            try:
                # User adoption indicators
                stars = row.get('stars', 0)
                dependents = row.get('dependents_count', 0)
                downloads = (row.get('pypi_downloads', 0) + row.get('npm_downloads', 0) + 
                           row.get('cargo_downloads', 0))
                
                # Adoption score (log scale)
                adoption_score = min(np.log(stars + dependents + downloads/100 + 1) / np.log(1000), 0.4)
                
                # Development maturity
                total_releases = row.get('total_releases', 0)
                maturity_score = min(total_releases / 10, 0.3)  # 10 releases = mature
                
                # Community engagement
                issues = row.get('issues_30_days', 0)
                prs = row.get('prs_30_days', 0)
                engagement_score = min((issues + prs) / 20, 0.2)  # Active community
                
                # Documentation and examples
                readme_quality = row.get('readme_quality_score', 0)
                doc_score = readme_quality * 0.1
                
                total_readiness = adoption_score + maturity_score + engagement_score + doc_score
                readiness_scores.append(min(total_readiness, 1.0))
                
            except Exception as e:
                logger.warning(f"Error calculating market readiness: {e}")
                readiness_scores.append(0)
        
        return pd.Series(readiness_scores)
    
    def _calculate_market_positioning_composite(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite market positioning score"""
        weights = {
            'problem_ambition_score': 0.3,
            'commercial_viability_score': 0.25,
            'technology_differentiation_score': 0.25,
            'market_readiness_score': 0.2
        }
        
        composite_scores = []
        for _, row in df.iterrows():
            weighted_sum = sum(row.get(signal, 0) * weight 
                             for signal, weight in weights.items())
            composite_scores.append(weighted_sum)
        
        return pd.Series(composite_scores)
    
    def calculate_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtask 2.5: Calculate composite scores and normalize features
        Final scoring and normalization of all engineered signals
        """
        logger.info("Calculating composite scores and normalization...")
        
        # Calculate overall category-defining potential score
        df['category_potential_score'] = self._calculate_category_potential(df)
        
        # Calculate funding-adjusted score (BSV-specific)
        df['bsv_investment_score'] = self._calculate_bsv_investment_score(df)
        
        # Normalize all engineered features to [0,1] scale
        df = self._normalize_engineered_features(df)
        
        # Calculate feature importance weights
        feature_importance = self._calculate_feature_importance(df)
        
        logger.info("✅ Composite scoring and normalization complete")
        return df, feature_importance
    
    def _calculate_category_potential(self, df: pd.DataFrame) -> pd.Series:
        """Calculate overall category-defining potential"""
        # Weights based on BSV's investment criteria
        weights = {
            'execution_velocity_composite': 0.25,
            'team_community_composite': 0.25,
            'technical_maturity_composite': 0.20,
            'market_positioning_composite': 0.30
        }
        
        potential_scores = []
        for _, row in df.iterrows():
            weighted_sum = sum(row.get(signal, 0) * weight 
                             for signal, weight in weights.items())
            potential_scores.append(weighted_sum)
        
        return pd.Series(potential_scores)
    
    def _calculate_bsv_investment_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate BSV-specific investment score with funding gate"""
        investment_scores = []
        
        for _, row in df.iterrows():
            try:
                # Base category potential
                category_potential = row.get('category_potential_score', 0)
                
                # Funding risk adjustment
                funding_confidence = row.get('funding_confidence', 0)
                funding_risk = row.get('funding_risk_level', 'uncertain_funding_status')
                
                # Apply funding gate (favor unfunded companies)
                if funding_risk == 'low_risk_unfunded':
                    funding_multiplier = 1.0
                elif funding_risk == 'uncertain_funding_status':
                    funding_multiplier = 0.8
                elif funding_risk == 'medium_risk_funded':
                    funding_multiplier = 0.6
                else:  # high_risk_funded
                    funding_multiplier = 0.4
                
                # BSV investment score
                bsv_score = category_potential * funding_multiplier
                investment_scores.append(bsv_score)
                
            except Exception as e:
                logger.warning(f"Error calculating BSV investment score: {e}")
                investment_scores.append(0)
        
        return pd.Series(investment_scores)
    
    def _normalize_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all engineered features to [0,1] scale"""
        # Identify engineered features (those with '_score' or '_composite' in name)
        engineered_features = [col for col in df.columns 
                              if '_score' in col or '_composite' in col]
        
        df_normalized = df.copy()
        
        for feature in engineered_features:
            try:
                values = df_normalized[feature].values
                if values.max() > values.min():  # Avoid division by zero
                    # Use robust min-max scaling (use 5th and 95th percentiles)
                    min_val = np.percentile(values, 5)
                    max_val = np.percentile(values, 95)
                    normalized = np.clip((values - min_val) / (max_val - min_val), 0, 1)
                    df_normalized[feature] = normalized
                else:
                    df_normalized[feature] = 0  # All values are the same
                    
            except Exception as e:
                logger.warning(f"Error normalizing {feature}: {e}")
        
        return df_normalized
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance based on variance and correlations"""
        engineered_features = [col for col in df.columns 
                              if '_score' in col or '_composite' in col]
        
        feature_importance = {}
        
        for feature in engineered_features:
            try:
                values = df[feature].values
                
                # Importance based on variance (features with more variance are more informative)
                variance_score = np.var(values)
                
                # Importance based on non-zero values (features that differentiate repositories)
                non_zero_ratio = (values > 0.01).mean()  # Ratio of non-negligible values
                
                # Combined importance
                importance = variance_score * non_zero_ratio
                feature_importance[feature] = importance
                
            except Exception as e:
                logger.warning(f"Error calculating importance for {feature}: {e}")
                feature_importance[feature] = 0
        
        # Normalize importance scores
        max_importance = max(feature_importance.values()) if feature_importance else 1
        if max_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] /= max_importance
        
        return feature_importance
    
    def _calculate_repo_age_months(self, created_at_str: str) -> float:
        """Calculate repository age in months"""
        try:
            if pd.isna(created_at_str):
                return 0
            
            from datetime import datetime
            created_date = pd.to_datetime(created_at_str, utc=True)
            now = pd.Timestamp.now(tz='UTC')
            age_days = (now - created_date).days
            return max(age_days / 30.44, 0.1)  # Convert to months, minimum 0.1
        except Exception:
            return 1.0  # Default to 1 month if parsing fails
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Apply all subtasks
        processed_df = self.engineer_execution_velocity_signals(processed_df)
        processed_df = self.engineer_team_community_signals(processed_df)  
        processed_df = self.engineer_technical_maturity_signals(processed_df)
        processed_df = self.engineer_market_positioning_signals(processed_df)
        
        # Final composite scoring and normalization
        processed_df, feature_importance = self.calculate_composite_scores(processed_df)
        
        logger.info(f"✅ Feature engineering complete: {len(processed_df.columns)} total features")
        
        return processed_df, feature_importance
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str, feature_importance: Dict[str, float] = None):
        """Save processed data with metadata"""
        try:
            # Save main data
            df.to_csv(output_path, index=False)
            
            # Save feature metadata
            metadata_path = output_path.replace('.csv', '_metadata.json')
            metadata = {
                'total_features': len(df.columns),
                'engineered_features': [col for col in df.columns if '_score' in col or '_composite' in col],
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'feature_categories': {
                    'execution_velocity': [col for col in df.columns if 'execution' in col or 'velocity' in col or 'commit' in col or 'release' in col or 'maintenance' in col or 'consistency' in col],
                    'team_community': [col for col in df.columns if 'team' in col or 'community' in col or 'growth' in col or 'network' in col or 'resilience' in col],
                    'technical_maturity': [col for col in df.columns if 'operational' in col or 'quality' in col or 'stability' in col or 'documentation' in col or 'maturity' in col],
                    'market_positioning': [col for col in df.columns if 'problem' in col or 'commercial' in col or 'differentiation' in col or 'market' in col or 'positioning' in col],
                    'composite_scores': [col for col in df.columns if 'category_potential' in col or 'bsv_investment' in col]
                },
                'feature_importance': feature_importance or {}
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Processed data saved to {output_path}")
            logger.info(f"✅ Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise

if __name__ == "__main__":
    # Test the feature engineering pipeline
    engineer = FeatureEngineer()
    
    # Load test data
    test_data_path = Path(__file__).parent.parent / "data" / "task1_complete_test.csv"
    if test_data_path.exists():
        df = engineer.load_data(str(test_data_path))
        processed_df, feature_importance = engineer.process_features(df)
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "task2_engineered_features.csv"
        engineer.save_processed_data(processed_df, str(output_path), feature_importance)
        
        print(f"✅ Feature engineering complete!")
        print(f"📊 Engineered {len(processed_df.columns)} features from {len(df.columns)} raw features")
        print(f"📁 Results saved to {output_path}")
    else:
        print("❌ Test data not found. Run Task 1 first to generate test data.")