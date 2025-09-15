#!/usr/bin/env python3
"""
Quick script to generate final enhanced rankings using the fixed final scorer
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from final_scorer import FinalScorer

def main():
    print("🚀 Generating Enhanced BSV Rankings with Fixed Final Scorer")
    print("=" * 60)
    
    # Load the data
    features_df = pd.read_csv('data/final_engineered_features.csv')
    llm_df = pd.read_csv('data/complete_real_dataset_llm_rankings.csv')
    
    print(f"📊 Loaded {len(features_df)} repositories with {len(features_df.columns)} engineered features")
    print(f"📊 Loaded {len(llm_df)} LLM rankings with real preference scores")
    print(f"   LLM scores range: {llm_df['llm_preference_score'].min():.3f} to {llm_df['llm_preference_score'].max():.3f}")
    
    # Initialize enhanced final scorer
    scorer = FinalScorer()
    print(f"\n⚙️  BSV Scoring Weights: Technical={scorer.weights.technical_execution:.0%}, LLM={scorer.weights.llm_preference:.0%}, Market={scorer.weights.market_adoption:.0%}, Team={scorer.weights.team_resilience:.0%}")
    
    # Calculate final scores using enhanced methods
    results_df = scorer.calculate_final_scores(features_df, llm_df)
    
    # Save results
    output_path = "output/bsv_enhanced_final_rankings.csv"
    results_df.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n📈 RESULTS SUMMARY")
    print(f"   Repositories scored: {len(results_df)}")
    print(f"   Final scores range: {results_df['final_score'].min():.3f} to {results_df['final_score'].max():.3f}")
    print(f"   Technical scores range: {results_df['technical_execution_score'].min():.3f} to {results_df['technical_execution_score'].max():.3f}")
    print(f"   BSV investment scores range: {results_df['bsv_investment_score'].min():.3f} to {results_df['bsv_investment_score'].max():.3f}")
    print(f"   Innovation scores range: {results_df['innovation_score'].min():.3f} to {results_df['innovation_score'].max():.3f}")
    
    print(f"\n🏆 TOP 10 BSV INVESTMENT TARGETS:")
    print("=" * 60)
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['repo_name']:<30} | Final: {row['final_score']:.3f} | Tech: {row['technical_execution_score']:.3f} | BSV: {row['bsv_investment_score']:.3f}")
    
    print(f"\n📁 Results saved to: {output_path}")
    print("✅ Enhanced BSV analysis complete!")

if __name__ == "__main__":
    main()