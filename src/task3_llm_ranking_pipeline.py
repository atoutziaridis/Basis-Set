"""
Task 3: LLM Pairwise Ranking System - Complete Pipeline
Integrates repository card generation, pair selection, LLM judging, and Bradley-Terry ranking
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import os

# Import Task 3 components
from .repository_card_generator import RepositoryCardGenerator
from .pair_selector import StrategicPairSelector
from .llm_judge import LLMJudge
from .bradley_terry_ranker import BradleyTerryRanker, create_integrated_rankings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMRankingPipeline:
    """Complete LLM-based repository ranking pipeline"""
    
    def __init__(self, target_comparisons: int = 300, 
                 openai_model: str = "gpt-3.5-turbo",
                 output_dir: Optional[str] = None):
        """Initialize pipeline components
        
        Args:
            target_comparisons: Number of pairwise comparisons to perform
            openai_model: OpenAI model to use for judgments
            output_dir: Output directory for results
        """
        
        self.target_comparisons = target_comparisons
        self.openai_model = openai_model
        
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent / "data"
        else:
            self.output_dir = Path(output_dir)
        
        # Initialize components
        self.card_generator = RepositoryCardGenerator(token_target=450)
        self.pair_selector = StrategicPairSelector(target_comparisons=target_comparisons)
        self.llm_judge = None  # Initialize when API key available
        self.bt_ranker = BradleyTerryRanker()
        
        # Pipeline state
        self.repository_cards = {}
        self.selected_pairs = []
        self.judgment_results = []
        self.final_rankings = None
        
    def run_complete_pipeline(self, input_csv_path: str, 
                            save_intermediate: bool = True) -> pd.DataFrame:
        """Run the complete LLM ranking pipeline
        
        Args:
            input_csv_path: Path to Task 2 engineered features CSV
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Final integrated rankings dataframe
        """
        
        logger.info("🚀 Starting LLM Ranking Pipeline (Task 3)")
        start_time = time.time()
        
        try:
            # Step 1: Load data
            logger.info("📂 Loading repository data...")
            df = pd.read_csv(input_csv_path)
            logger.info(f"Loaded {len(df)} repositories")
            
            # Step 2: Generate repository cards
            logger.info("📋 Generating repository cards...")
            self.repository_cards = self.card_generator.generate_cards_batch(df)
            
            if save_intermediate:
                cards_path = self.output_dir / "task3_repository_cards.json"
                self.card_generator.save_cards(self.repository_cards, str(cards_path))
            
            # Step 3: Select strategic pairs
            logger.info("🎯 Selecting strategic pairs...")
            self.selected_pairs = self.pair_selector.select_pairs(df, 'category_potential_score')
            
            if save_intermediate:
                pairs_path = self.output_dir / "task3_selected_pairs.json"
                self.pair_selector.save_pairs(self.selected_pairs, str(pairs_path), df)
            
            # Step 4: LLM Judging
            logger.info("🧠 Performing LLM pairwise judgments...")
            self.judgment_results = self._perform_llm_judging(df)
            
            if save_intermediate:
                judgments_path = self.output_dir / "task3_llm_judgments.json"
                self._save_judgments(self.judgment_results, str(judgments_path))
            
            # Step 5: Bradley-Terry Ranking
            logger.info("🏆 Aggregating rankings with Bradley-Terry model...")
            repo_names = df['repo_name'].tolist()
            self.bt_ranker.fit(self.judgment_results, repo_names)
            bt_rankings = self.bt_ranker.get_rankings()
            
            if save_intermediate:
                bt_path = self.output_dir / "task3_bradley_terry_rankings.json"
                self.bt_ranker.save_rankings(str(bt_path))
            
            # Step 6: Create integrated final rankings
            logger.info("📊 Creating integrated final rankings...")
            self.final_rankings = create_integrated_rankings(bt_rankings, df)
            
            # Save final results
            final_path = self.output_dir / "task3_final_llm_rankings.csv"
            self.final_rankings.to_csv(str(final_path), index=False)
            
            # Save detailed results with metadata
            self._save_final_results(df)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ LLM Ranking Pipeline completed in {elapsed_time:.1f}s")
            
            # Print summary
            self._print_pipeline_summary()
            
            return self.final_rankings
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def _perform_llm_judging(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform LLM pairwise judgments
        
        Args:
            df: Repository dataframe
            
        Returns:
            List of judgment results
        """
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("⚠️  No OPENAI_API_KEY found. Using mock judgments for testing.")
            return self._generate_mock_judgments(df)
        
        # Initialize LLM judge
        self.llm_judge = LLMJudge(model=self.openai_model)
        
        # Perform judgments
        repo_names = df['repo_name'].tolist()
        results = self.llm_judge.judge_pairs_batch(
            self.selected_pairs, 
            self.repository_cards, 
            repo_names
        )
        
        # Convert results to dictionaries
        judgment_dicts = []
        for i, result in enumerate(results):
            idx1, idx2 = self.selected_pairs[i]
            repo1_name = repo_names[idx1] if idx1 < len(repo_names) else f"repo_{idx1}"
            repo2_name = repo_names[idx2] if idx2 < len(repo_names) else f"repo_{idx2}"
            
            judgment_dicts.append({
                'pair_id': i,
                'repo1_name': repo1_name,
                'repo2_name': repo2_name,
                'winner': result.winner,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'processing_time': result.processing_time,
                'raw_response': result.raw_response
            })
        
        return judgment_dicts
    
    def _generate_mock_judgments(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate mock judgments for testing when API key unavailable
        
        Args:
            df: Repository dataframe
            
        Returns:
            List of mock judgment results
        """
        
        logger.info("Generating mock judgments for testing...")
        
        import random
        
        mock_judgments = []
        repo_names = df['repo_name'].tolist()
        
        for i, (idx1, idx2) in enumerate(self.selected_pairs):
            repo1_name = repo_names[idx1] if idx1 < len(repo_names) else f"repo_{idx1}"
            repo2_name = repo_names[idx2] if idx2 < len(repo_names) else f"repo_{idx2}"
            
            # Use category potential scores to inform mock judgment
            score1 = df.iloc[idx1].get('category_potential_score', 0.5)
            score2 = df.iloc[idx2].get('category_potential_score', 0.5)
            
            # Bias toward higher scoring repository
            prob_repo1_wins = score1 / (score1 + score2) if (score1 + score2) > 0 else 0.5
            winner = 0 if random.random() < prob_repo1_wins else 1
            
            mock_judgments.append({
                'pair_id': i,
                'repo1_name': repo1_name,
                'repo2_name': repo2_name,
                'winner': winner,
                'confidence': random.choice(['low', 'medium', 'high']),
                'reasoning': f"Mock judgment based on feature scores",
                'processing_time': random.uniform(0.5, 2.0),
                'raw_response': "MOCK_RESPONSE"
            })
        
        return mock_judgments
    
    def _save_judgments(self, judgments: List[Dict[str, Any]], output_path: str):
        """Save judgment results to file
        
        Args:
            judgments: List of judgment dictionaries
            output_path: Output file path
        """
        
        try:
            output_data = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'total_judgments': len(judgments),
                'model_used': self.openai_model,
                'target_comparisons': self.target_comparisons,
                'judgments': judgments
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"✅ Saved {len(judgments)} judgments to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save judgments: {e}")
    
    def _save_final_results(self, original_df: pd.DataFrame):
        """Save comprehensive final results with metadata
        
        Args:
            original_df: Original features dataframe
        """
        
        try:
            # Prepare comprehensive results
            results_data = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'pipeline_config': {
                    'target_comparisons': self.target_comparisons,
                    'llm_model': self.openai_model,
                    'total_repositories': len(original_df)
                },
                'pipeline_stats': {
                    'cards_generated': len(self.repository_cards),
                    'pairs_selected': len(self.selected_pairs),
                    'judgments_completed': len(self.judgment_results),
                    'final_rankings': len(self.final_rankings)
                },
                'model_diagnostics': self.bt_ranker.get_model_diagnostics() if self.bt_ranker.scores_ is not None else {},
                'pair_selection_report': self.pair_selector.get_coverage_report(),
                'top_10_repositories': self.final_rankings.head(10).to_dict('records'),
                'rankings_summary': {
                    'total_ranked': len(self.final_rankings),
                    'score_range': {
                        'min': float(self.final_rankings['integrated_score'].min()),
                        'max': float(self.final_rankings['integrated_score'].max()),
                        'mean': float(self.final_rankings['integrated_score'].mean())
                    }
                }
            }
            
            # Add LLM usage stats if available
            if self.llm_judge is not None:
                results_data['llm_usage_stats'] = self.llm_judge.get_usage_stats()
            
            # Save comprehensive results
            results_path = self.output_dir / "task3_complete_results.json"
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"✅ Saved comprehensive results to {results_path}")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
    
    def _print_pipeline_summary(self):
        """Print pipeline execution summary"""
        
        print("\n" + "="*60)
        print("🏆 LLM RANKING PIPELINE - EXECUTION SUMMARY")
        print("="*60)
        
        print(f"📋 Repository Cards Generated: {len(self.repository_cards)}")
        print(f"🎯 Strategic Pairs Selected: {len(self.selected_pairs)}")
        print(f"🧠 LLM Judgments Completed: {len(self.judgment_results)}")
        
        if self.final_rankings is not None:
            print(f"🏆 Final Rankings Created: {len(self.final_rankings)}")
            print(f"\n🥇 TOP 10 REPOSITORIES:")
            print("-"*40)
            for idx, row in self.final_rankings.head(10).iterrows():
                rank = int(row['final_rank'])
                repo = row['repository']
                score = row['integrated_score']
                bt_score = row['bradley_terry_score']
                print(f"  {rank:2d}. {repo:<30} (Score: {score:.4f}, BT: {bt_score:.4f})")
        
        # Print diagnostics if available
        if self.bt_ranker.scores_ is not None:
            diagnostics = self.bt_ranker.get_model_diagnostics()
            print(f"\n📊 Bradley-Terry Model:")
            print(f"  Converged: {diagnostics['convergence_status']['converged']}")
            print(f"  Iterations: {diagnostics['convergence_status']['iterations']}")
            print(f"  Coverage: {diagnostics['comparison_coverage']:.1%}")
        
        # Print LLM usage if available
        if self.llm_judge is not None:
            stats = self.llm_judge.get_usage_stats()
            print(f"\n💰 LLM Usage:")
            print(f"  Requests: {stats['total_requests']}")
            print(f"  Tokens: {stats['total_tokens']:,}")
            print(f"  Est. Cost: ${stats['estimated_cost_usd']:.2f}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
        
        print("\n✅ Pipeline completed successfully!")
        print("="*60)

    def load_previous_results(self, results_dir: str = None) -> pd.DataFrame:
        """Load previously generated rankings
        
        Args:
            results_dir: Directory containing results
            
        Returns:
            Previously generated rankings
        """
        
        if results_dir is None:
            results_dir = self.output_dir
        
        results_path = Path(results_dir) / "task3_final_llm_rankings.csv"
        
        if results_path.exists():
            rankings = pd.read_csv(str(results_path))
            logger.info(f"✅ Loaded previous rankings from {results_path}")
            return rankings
        else:
            logger.error(f"No previous results found at {results_path}")
            return None

def run_task3_pipeline(input_csv_path: str, 
                      target_comparisons: int = 200,
                      openai_model: str = "gpt-3.5-turbo") -> pd.DataFrame:
    """Convenience function to run Task 3 pipeline
    
    Args:
        input_csv_path: Path to Task 2 engineered features CSV
        target_comparisons: Number of pairwise comparisons
        openai_model: OpenAI model to use
        
    Returns:
        Final rankings dataframe
    """
    
    pipeline = LLMRankingPipeline(
        target_comparisons=target_comparisons,
        openai_model=openai_model
    )
    
    return pipeline.run_complete_pipeline(input_csv_path)

if __name__ == "__main__":
    # Run Task 3 pipeline
    print("🚀 Running Task 3: LLM Pairwise Ranking Pipeline")
    
    # Input file path
    data_dir = Path(__file__).parent.parent / "data"
    input_path = data_dir / "task2_engineered_features.csv"
    
    if not input_path.exists():
        print("❌ Task 2 engineered features not found.")
        print("   Please run Task 2 first to generate the required input data.")
        exit()
    
    try:
        # Create and run pipeline
        pipeline = LLMRankingPipeline(
            target_comparisons=50,  # Small number for testing
            openai_model="gpt-3.5-turbo"
        )
        
        # Run complete pipeline
        final_rankings = pipeline.run_complete_pipeline(str(input_path))
        
        print(f"\n🎉 Task 3 completed successfully!")
        print(f"📊 Generated rankings for {len(final_rankings)} repositories")
        print(f"💾 Results saved to {data_dir}")
        
    except Exception as e:
        print(f"❌ Task 3 pipeline failed: {e}")
        import traceback
        traceback.print_exc()