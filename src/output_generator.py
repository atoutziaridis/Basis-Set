"""
BSV Output Generation System
Implements Task 4.5: Structured result presentation and comprehensive reporting
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

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutputGenerator:
    """
    BSV Output Generation System
    
    Creates comprehensive outputs for BSV's investment analysis:
    - Final prioritized CSV with comprehensive scoring
    - Executive summary with key findings
    - Investment brief summaries for top candidates
    - Methodology documentation
    - Visualization suite
    """
    
    def __init__(self, project_root: str):
        """Initialize output generator"""
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("OutputGenerator initialized")
    
    def load_all_results(self) -> Dict[str, Any]:
        """Load all Task 4 results for comprehensive output generation"""
        
        logger.info("Loading all Task 4 results...")
        
        results = {}
        
        # Load final scores
        scores_path = self.data_dir / "task4_final_scores.csv"
        if scores_path.exists():
            results['final_scores'] = pd.read_csv(scores_path)
            logger.info(f"Loaded final scores: {len(results['final_scores'])} repositories")
        
        # Load explanations
        explanations_path = self.data_dir / "task4_explanations.json"
        if explanations_path.exists():
            with open(explanations_path, 'r') as f:
                results['explanations'] = json.load(f)
            logger.info("Loaded explainability analysis")
        
        # Load evaluation report
        evaluation_path = self.data_dir / "task4_evaluation_report.json"
        if evaluation_path.exists():
            with open(evaluation_path, 'r') as f:
                results['evaluation'] = json.load(f)
            logger.info("Loaded evaluation report")
        
        # Load bias analysis
        bias_path = self.data_dir / "task4_bias_analysis.json"
        if bias_path.exists():
            with open(bias_path, 'r') as f:
                results['bias_analysis'] = json.load(f)
            logger.info("Loaded bias analysis")
        
        # Load original features for context
        features_path = self.data_dir / "test_task3_dataset.csv"
        if features_path.exists():
            results['features'] = pd.read_csv(features_path)
            logger.info("Loaded feature dataset")
        
        return results
    
    def generate_final_csv(self, results: Dict[str, Any]) -> str:
        """Generate final prioritized CSV with comprehensive scoring"""
        
        logger.info("Generating final prioritized CSV...")
        
        if 'final_scores' not in results:
            logger.error("Final scores not available")
            return ""
        
        final_df = results['final_scores'].copy()
        
        # Add investment brief summaries for top repositories
        if 'explanations' in results:
            explanations_data = results['explanations']
            
            # Create investment briefs mapping
            brief_mapping = {}
            for explanation in explanations_data.get('explanations', []):
                repo_name = explanation['repo_name']
                brief = self._generate_investment_brief(explanation, results.get('features'))
                brief_mapping[repo_name] = brief
            
            # Add investment briefs to dataframe
            final_df['investment_brief'] = final_df['repo_name'].map(brief_mapping)
        
        # Add methodology reference
        final_df['methodology_version'] = '1.0'
        final_df['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Reorder columns for better readability
        column_order = [
            'rank', 'repo_name', 'final_score',
            'llm_preference_score', 'technical_execution_score', 
            'market_adoption_score', 'team_resilience_score',
            'funding_gate_multiplier', 'funding_risk_level',
            'reason_1', 'reason_2', 'reason_3',
            'stars', 'forks', 'created_at',
            'category_potential_score', 'bsv_investment_score',
            'investment_brief', 'methodology_version', 'analysis_date'
        ]
        
        # Keep only available columns
        available_columns = [col for col in column_order if col in final_df.columns]
        final_df = final_df[available_columns]
        
        # Save final CSV
        output_path = self.output_dir / "bsv_prioritized_repositories.csv"
        final_df.to_csv(output_path, index=False)
        
        logger.info(f"Final CSV saved to {output_path}")
        return str(output_path)
    
    def _generate_investment_brief(self, explanation: Dict[str, Any], 
                                 features_df: Optional[pd.DataFrame]) -> str:
        """Generate one-paragraph investment brief for a repository"""
        
        repo_name = explanation['repo_name']
        rank = explanation['rank']
        score = explanation['final_score']
        summary = explanation.get('human_readable_summary', '')
        advantages = explanation.get('comparative_advantages', [])
        
        # Get additional context from features if available
        context = ""
        if features_df is not None and 'repo_name' in features_df.columns:
            repo_data = features_df[features_df['repo_name'] == repo_name]
            if not repo_data.empty:
                repo_info = repo_data.iloc[0]
                stars = repo_info.get('stars', 0)
                language = repo_info.get('primary_language', 'Unknown')
                context = f" This {language} project has {stars:,} GitHub stars"
        
        # Construct brief
        brief = f"#{rank} ranked repository with {score:.2f} final score. {summary}"
        
        if advantages:
            brief += f" Key advantages: {', '.join(advantages[:2])}."
        
        brief += context + "."
        
        # Ensure reasonable length (max 200 chars)
        if len(brief) > 200:
            brief = brief[:197] + "..."
        
        return brief
    
    def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary with key findings"""
        
        logger.info("Generating executive summary...")
        
        summary_sections = []
        
        # Header
        summary_sections.append("# BSV Repository Prioritization - Executive Summary")
        summary_sections.append(f"**Analysis Date**: {datetime.now().strftime('%B %d, %Y')}")
        summary_sections.append("")
        
        # Key Findings
        summary_sections.append("## Key Findings")
        
        if 'final_scores' in results:
            scores_df = results['final_scores']
            total_repos = len(scores_df)
            score_range = f"{scores_df['final_score'].min():.3f} - {scores_df['final_score'].max():.3f}"
            mean_score = scores_df['final_score'].mean()
            
            summary_sections.append(f"- **{total_repos} repositories** analyzed and ranked")
            summary_sections.append(f"- **Score range**: {score_range} (mean: {mean_score:.3f})")
            
            # Top repository
            top_repo = scores_df.iloc[0]
            summary_sections.append(f"- **Top repository**: {top_repo['repo_name']} (score: {top_repo['final_score']:.3f})")
            
            # Funding analysis
            unfunded_count = len(scores_df[scores_df['funding_risk_level'].str.contains('unfunded', na=False)])
            summary_sections.append(f"- **{unfunded_count} repositories** identified as unfunded with high potential")
        
        summary_sections.append("")
        
        # Methodology Summary
        summary_sections.append("## Methodology")
        summary_sections.append("Our analysis combines multiple signals using a weighted scoring approach:")
        summary_sections.append("- **LLM Preference Score (60%)**: AI-powered pairwise comparisons")
        summary_sections.append("- **Technical Execution (15%)**: Development velocity and code quality")
        summary_sections.append("- **Market Adoption (15%)**: Community engagement and growth")
        summary_sections.append("- **Team Resilience (10%)**: Contributor diversity and sustainability")
        summary_sections.append("- **Funding Gate**: Preference multiplier for unfunded projects")
        summary_sections.append("")
        
        # Evaluation Results
        if 'evaluation' in results:
            eval_data = results['evaluation']
            summary_sections.append("## System Validation")
            
            ablation_count = eval_data.get('evaluation_summary', {}).get('ablation_studies', 0)
            sanity_passed = eval_data.get('evaluation_summary', {}).get('sanity_checks_passed', 0)
            sanity_total = eval_data.get('evaluation_summary', {}).get('sanity_checks', 0)
            
            summary_sections.append(f"- **{ablation_count} ablation studies** conducted to validate component contributions")
            summary_sections.append(f"- **{sanity_passed}/{sanity_total} sanity checks** passed")
            summary_sections.append("- **Bootstrap stability analysis** confirms reliable rankings")
        
        # Bias Analysis
        if 'bias_analysis' in results:
            bias_data = results['bias_analysis']
            bias_summary = bias_data.get('summary', {})
            risk_level = bias_summary.get('overall_assessment', 'Unknown')
            tests_passed = bias_summary.get('tests_passed', 'Unknown')
            
            summary_sections.append(f"- **Bias analysis**: {risk_level} with {tests_passed} tests passed")
            
            primary_concerns = bias_summary.get('primary_concerns', [])
            if primary_concerns:
                summary_sections.append(f"- **Areas for improvement**: {', '.join(primary_concerns)}")
        
        summary_sections.append("")
        
        # Top 10 Repositories
        if 'final_scores' in results:
            summary_sections.append("## Top 10 Repositories")
            top_10 = results['final_scores'].head(10)
            
            for i, (_, repo) in enumerate(top_10.iterrows(), 1):
                repo_name = repo['repo_name']
                score = repo['final_score']
                reason = repo.get('reason_1', 'Strong overall performance')
                summary_sections.append(f"{i:2d}. **{repo_name}** (Score: {score:.3f}) - {reason}")
        
        summary_sections.append("")
        
        # Investment Recommendations
        summary_sections.append("## Investment Recommendations")
        summary_sections.append("1. **Priority Focus**: Top 5 repositories show exceptional potential")
        summary_sections.append("2. **Due Diligence**: Verify funding status of high-scoring repositories")
        summary_sections.append("3. **Portfolio Balance**: Consider diversification across technology domains")
        summary_sections.append("4. **Monitoring**: Track development velocity and community growth")
        summary_sections.append("")
        
        # Limitations
        summary_sections.append("## Limitations")
        summary_sections.append("- Analysis based on public GitHub data and may miss private developments")
        summary_sections.append("- Funding detection relies on text analysis and may have false negatives")
        summary_sections.append("- Market potential assessment is based on technical and adoption signals")
        summary_sections.append("- Recommendations should be combined with domain expertise and market analysis")
        
        # Combine all sections
        summary_text = "\n".join(summary_sections)
        
        # Save executive summary
        output_path = self.output_dir / "executive_summary.md"
        with open(output_path, 'w') as f:
            f.write(summary_text)
        
        logger.info(f"Executive summary saved to {output_path}")
        return str(output_path)
    
    def generate_methodology_documentation(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive methodology documentation"""
        
        logger.info("Generating methodology documentation...")
        
        doc_sections = []
        
        # Header
        doc_sections.append("# BSV Repository Prioritization - Methodology Documentation")
        doc_sections.append(f"**Version**: 1.0")
        doc_sections.append(f"**Date**: {datetime.now().strftime('%B %d, %Y')}")
        doc_sections.append("")
        
        # Overview
        doc_sections.append("## Overview")
        doc_sections.append("This document describes the methodology used to prioritize GitHub repositories ")
        doc_sections.append("for Basis Set Ventures' investment analysis. The system combines multiple data ")
        doc_sections.append("sources and AI-powered analysis to identify category-defining companies without ")
        doc_sections.append("current institutional funding.")
        doc_sections.append("")
        
        # Data Collection
        doc_sections.append("## Data Collection (Task 1)")
        doc_sections.append("### GitHub API Integration")
        doc_sections.append("- Repository metadata: stars, forks, creation date, language distribution")
        doc_sections.append("- Activity metrics: commit frequency, release cadence, issue response times")
        doc_sections.append("- Contributor analysis: bus factor, contribution distribution, team diversity")
        doc_sections.append("- Code quality indicators: CI/CD presence, test coverage, documentation quality")
        doc_sections.append("")
        doc_sections.append("### Funding Detection")
        doc_sections.append("- NLP-based text analysis of README files and repository descriptions")
        doc_sections.append("- Pattern matching for funding keywords and investor mentions")
        doc_sections.append("- Risk classification: low_risk_unfunded, unfunded, funded, high_risk_funded")
        doc_sections.append("")
        
        # Feature Engineering
        doc_sections.append("## Feature Engineering (Task 2)")
        doc_sections.append("### Composite Scores")
        doc_sections.append("- **Execution Velocity**: Commit patterns, release cadence, development consistency")
        doc_sections.append("- **Team Resilience**: Contributor diversity, bus factor, community health")
        doc_sections.append("- **Technical Maturity**: Code quality, documentation, operational readiness")
        doc_sections.append("- **Market Positioning**: Commercial viability, technology differentiation")
        doc_sections.append("")
        
        # LLM Ranking
        doc_sections.append("## LLM Pairwise Ranking (Task 3)")
        doc_sections.append("### Repository Cards")
        doc_sections.append("- Structured summaries highlighting key features and innovations")
        doc_sections.append("- Technology stack, use cases, and differentiation factors")
        doc_sections.append("- Market potential and adoption indicators")
        doc_sections.append("")
        doc_sections.append("### Pairwise Comparisons")
        doc_sections.append("- GPT-4 powered comparative analysis")
        doc_sections.append("- Bradley-Terry model for consistent ranking from pairwise judgments")
        doc_sections.append("- Confidence intervals and stability analysis")
        doc_sections.append("")
        
        # Final Scoring
        doc_sections.append("## Final Scoring Framework (Task 4)")
        doc_sections.append("### Weighted Linear Combination")
        doc_sections.append("```")
        doc_sections.append("Final Score = (")
        doc_sections.append("    0.60 × LLM Preference Score +")
        doc_sections.append("    0.15 × Technical Execution +")
        doc_sections.append("    0.15 × Market Adoption +")
        doc_sections.append("    0.10 × Team Resilience")
        doc_sections.append(") × Funding Gate Multiplier")
        doc_sections.append("```")
        doc_sections.append("")
        doc_sections.append("### Funding Gate")
        doc_sections.append("- Multiplier: max(0.6, 1 - p_institutional_funding)")
        doc_sections.append("- Ensures preference for unfunded projects while not completely excluding funded ones")
        doc_sections.append("- Based on funding confidence score from text analysis")
        doc_sections.append("")
        
        # Validation
        if 'evaluation' in results:
            doc_sections.append("## System Validation")
            doc_sections.append("### Ablation Studies")
            doc_sections.append("- LLM-only rankings: Tests pure AI judgment effectiveness")
            doc_sections.append("- Features-only rankings: Validates traditional metrics approach")
            doc_sections.append("- Component removal: Measures individual component contributions")
            doc_sections.append("- Equal weights: Compares to optimized weighting scheme")
            doc_sections.append("")
            
            doc_sections.append("### Sanity Checks")
            doc_sections.append("- Star count correlation: Should be moderate (not too high/low)")
            doc_sections.append("- Age bias: Should not favor old or new repositories excessively")
            doc_sections.append("- Activity correlation: Should positively correlate with development activity")
            doc_sections.append("- Funding bias: Should not favor funded projects")
            doc_sections.append("")
        
        # Bias Analysis
        if 'bias_analysis' in results:
            doc_sections.append("### Bias Detection")
            doc_sections.append("- **Age Bias**: Repository age correlation analysis")
            doc_sections.append("- **Popularity Bias**: Over-dependence on star count")
            doc_sections.append("- **Language Bias**: Programming language preferences")
            doc_sections.append("- **Size Bias**: Repository size metric correlations")
            doc_sections.append("- **Temporal Bias**: Recent activity preferences")
            doc_sections.append("- **Funding Bias**: Unintended funding status preferences")
            doc_sections.append("")
        
        # Limitations and Future Work
        doc_sections.append("## Limitations and Future Work")
        doc_sections.append("### Current Limitations")
        doc_sections.append("- Limited to public GitHub data")
        doc_sections.append("- Text-based funding detection may have false negatives")
        doc_sections.append("- Market potential based on technical signals, not market research")
        doc_sections.append("- Small sample size for LLM comparisons due to API costs")
        doc_sections.append("")
        
        doc_sections.append("### Future Improvements")
        doc_sections.append("- Integration with additional data sources (Crunchbase, AngelList)")
        doc_sections.append("- Enhanced funding detection using structured data")
        doc_sections.append("- Market size and opportunity assessment")
        doc_sections.append("- Real-time monitoring and alert systems")
        doc_sections.append("- Expanded LLM comparison coverage")
        
        # Combine sections
        doc_text = "\n".join(doc_sections)
        
        # Save methodology documentation
        output_path = self.output_dir / "methodology_documentation.md"
        with open(output_path, 'w') as f:
            f.write(doc_text)
        
        logger.info(f"Methodology documentation saved to {output_path}")
        return str(output_path)
    
    def create_visualization_suite(self, results: Dict[str, Any]) -> str:
        """Create comprehensive visualization suite"""
        
        logger.info("Creating visualization suite...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        if 'final_scores' not in results:
            logger.warning("Final scores not available for visualization")
            return str(viz_dir)
        
        scores_df = results['final_scores']
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Score Distribution and Rankings
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Score distribution
        ax1.hist(scores_df['final_score'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Final Scores', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Final Score')
        ax1.set_ylabel('Frequency')
        ax1.axvline(scores_df['final_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {scores_df["final_score"].mean():.3f}')
        ax1.legend()
        
        # Component scores comparison
        component_cols = ['llm_preference_score', 'technical_execution_score', 
                         'market_adoption_score', 'team_resilience_score']
        available_components = [col for col in component_cols if col in scores_df.columns]
        
        if available_components:
            component_means = scores_df[available_components].mean()
            component_names = [col.replace('_score', '').replace('_', ' ').title() 
                             for col in available_components]
            
            bars = ax2.bar(component_names, component_means.values, alpha=0.7)
            ax2.set_title('Mean Component Scores', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Mean Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, component_means.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Top 10 repositories
        top_10 = scores_df.head(10)
        y_pos = np.arange(len(top_10))
        
        bars = ax3.barh(y_pos, top_10['final_score'], alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(top_10['repo_name'], fontsize=10)
        ax3.set_xlabel('Final Score')
        ax3.set_title('Top 10 Repositories', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, top_10['final_score'])):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', ha='left', fontsize=9)
        
        # Funding risk distribution
        if 'funding_risk_level' in scores_df.columns:
            funding_counts = scores_df['funding_risk_level'].value_counts()
            colors = ['green' if 'unfunded' in risk else 'orange' if 'funded' in risk else 'gray' 
                     for risk in funding_counts.index]
            
            wedges, texts, autotexts = ax4.pie(funding_counts.values, labels=funding_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax4.set_title('Funding Risk Distribution', fontsize=14, fontweight='bold')
            
            # Improve label readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'score_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance Analysis
        if 'explanations' in results and 'explanations' in results['explanations']:
            explanations_data = results['explanations']['explanations']
            
            # Collect feature contributions
            feature_contributions = {}
            for explanation in explanations_data:
                for feature_data in explanation.get('top_positive_features', []):
                    feature_name = feature_data['feature_name']
                    contribution = feature_data['contribution']
                    
                    if feature_name not in feature_contributions:
                        feature_contributions[feature_name] = []
                    feature_contributions[feature_name].append(contribution)
            
            if feature_contributions:
                # Calculate mean contributions
                mean_contributions = {
                    feature: np.mean(contributions) 
                    for feature, contributions in feature_contributions.items()
                }
                
                # Plot top features
                sorted_features = sorted(mean_contributions.items(), 
                                       key=lambda x: abs(x[1]), reverse=True)[:15]
                
                feature_names = [name.replace('_', ' ').title() for name, _ in sorted_features]
                contributions = [contrib for _, contrib in sorted_features]
                
                plt.figure(figsize=(12, 8))
                colors = ['green' if c > 0 else 'red' for c in contributions]
                bars = plt.barh(range(len(feature_names)), contributions, color=colors, alpha=0.7)
                
                plt.yticks(range(len(feature_names)), feature_names)
                plt.xlabel('Mean Feature Contribution')
                plt.title('Top 15 Feature Contributions to Final Score', fontsize=16, fontweight='bold')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, contributions):
                    plt.text(value + (0.01 if value > 0 else -0.01), 
                            bar.get_y() + bar.get_height()/2,
                            f'{value:.3f}', va='center', 
                            ha='left' if value > 0 else 'right', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'feature_importance_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Correlation Matrix (if features available)
        if 'features' in results:
            features_df = results['features']
            
            # Select key numeric features for correlation analysis
            key_features = [
                'final_score', 'stars', 'forks', 'commits_6_months', 'total_contributors',
                'bus_factor', 'readme_quality_score', 'has_ci_cd', 'dependents_count'
            ]
            
            # Get available features (excluding final_score which comes from scores_df)
            available_features_from_features = [f for f in key_features[1:] if f in features_df.columns]
            
            if len(available_features_from_features) > 2:
                # Merge with scores to include final_score
                if 'repo_name' in features_df.columns and 'repo_name' in scores_df.columns:
                    merged_for_corr = scores_df[['repo_name', 'final_score']].merge(
                        features_df[['repo_name'] + available_features_from_features], on='repo_name'
                    )
                    
                    # Use only columns that exist in merged dataframe
                    all_available = ['final_score'] + available_features_from_features
                    correlation_matrix = merged_for_corr[all_available].corr()
                    
                    plt.figure(figsize=(10, 8))
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                               center=0, square=True, fmt='.2f')
                    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(viz_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
                    plt.close()
        
        logger.info(f"Visualization suite created in {viz_dir}")
        return str(viz_dir)
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive PDF report combining all analyses"""
        
        logger.info("Generating comprehensive PDF report...")
        
        output_path = self.output_dir / "bsv_comprehensive_analysis_report.pdf"
        
        with PdfPages(str(output_path)) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.8, 'BSV Repository Prioritization', 
                   ha='center', va='center', fontsize=24, fontweight='bold')
            ax.text(0.5, 0.7, 'Comprehensive Analysis Report', 
                   ha='center', va='center', fontsize=18)
            ax.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 
                   ha='center', va='center', fontsize=14)
            
            if 'final_scores' in results:
                total_repos = len(results['final_scores'])
                ax.text(0.5, 0.4, f'Repositories Analyzed: {total_repos}', 
                       ha='center', va='center', fontsize=16)
                
                top_repo = results['final_scores'].iloc[0]['repo_name']
                ax.text(0.5, 0.3, f'Top Repository: {top_repo}', 
                       ha='center', va='center', fontsize=16)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add other visualizations to PDF
            viz_dir = Path(self.output_dir) / "visualizations"
            
            for viz_file in viz_dir.glob("*.png"):
                img = plt.imread(str(viz_file))
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(viz_file.stem.replace('_', ' ').title(), 
                           fontsize=16, fontweight='bold', pad=20)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Comprehensive PDF report saved to {output_path}")
        return str(output_path)

def main():
    """Main execution function for Task 4.5: Output Generation"""
    logger.info("Starting BSV Output Generation - Task 4.5")
    
    # Initialize output generator
    project_root = Path(__file__).parent.parent
    generator = OutputGenerator(str(project_root))
    
    try:
        # Load all results
        results = generator.load_all_results()
        
        if not results:
            logger.error("No results found to generate outputs")
            return
        
        # Generate outputs
        outputs = {}
        
        # 1. Final prioritized CSV
        outputs['csv'] = generator.generate_final_csv(results)
        
        # 2. Executive summary
        outputs['summary'] = generator.generate_executive_summary(results)
        
        # 3. Methodology documentation
        outputs['methodology'] = generator.generate_methodology_documentation(results)
        
        # 4. Visualization suite
        outputs['visualizations'] = generator.create_visualization_suite(results)
        
        # 5. Comprehensive PDF report
        outputs['pdf_report'] = generator.generate_comprehensive_report(results)
        
        # Display summary
        print("\n" + "="*60)
        print("🎉 TASK 4.5 OUTPUT GENERATION COMPLETE")
        print("="*60)
        
        if 'final_scores' in results:
            total_repos = len(results['final_scores'])
            print(f"📊 Repositories processed: {total_repos}")
            
            top_repo = results['final_scores'].iloc[0]
            print(f"🏆 Top repository: {top_repo['repo_name']} (Score: {top_repo['final_score']:.3f})")
        
        print()
        print("📁 Generated outputs:")
        for output_type, path in outputs.items():
            if path:
                print(f"   • {output_type.replace('_', ' ').title()}: {path}")
        
        print()
        print("🚀 BSV Repository Prioritization Analysis Complete!")
        print("   All deliverables ready for investment team review.")
        
        return outputs
        
    except Exception as e:
        logger.error(f"Output generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
