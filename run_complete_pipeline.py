"""
Complete BSV Pipeline Runner
Executes Task 1 (Data Collection) + Task 2 (Feature Engineering) in sequence
"""

import sys
import time
from pathlib import Path
from src.data_collection_runner import main as run_task1
from src.feature_engineer import FeatureEngineer

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def run_complete_pipeline():
    """Run complete Task 1 + Task 2 pipeline"""
    print("🚀 BSV Repository Prioritizer - Complete Pipeline")
    print("=" * 60)
    print("This will run:")
    print("  ✅ Task 1: Data Collection & Enrichment (87 features)")
    print("  ✅ Task 2: Feature Engineering & Signals (22 features)")
    print("  📊 Total: 109 comprehensive features per repository")
    print()
    print("Expected processing time: ~45 minutes for 100 repositories")
    print()
    
    response = input("Run complete pipeline? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Pipeline cancelled.")
        return
    
    start_time = time.time()
    
    try:
        # Task 1: Data Collection
        print("\n" + "="*60)
        print("🔄 TASK 1: DATA COLLECTION & ENRICHMENT")
        print("="*60)
        
        task1_start = time.time()
        run_task1()
        task1_time = time.time() - task1_start
        
        print(f"✅ Task 1 completed in {task1_time/60:.1f} minutes")
        
        # Task 2: Feature Engineering
        print("\n" + "="*60)
        print("🔄 TASK 2: FEATURE ENGINEERING & SIGNALS")
        print("="*60)
        
        task2_start = time.time()
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Load Task 1 results
        task1_output = Path(__file__).parent / "data" / "enriched_github_data.csv"
        if not task1_output.exists():
            print("❌ Task 1 output not found. Task 1 may have failed.")
            return
        
        # Process features
        print("Loading Task 1 data...")
        df = engineer.load_data(str(task1_output))
        
        print("Engineering features...")
        processed_df, feature_importance = engineer.process_features(df)
        
        # Save results
        task2_output = Path(__file__).parent / "data" / "bsv_prioritized_features.csv"
        engineer.save_processed_data(processed_df, str(task2_output), feature_importance)
        
        task2_time = time.time() - task2_start
        print(f"✅ Task 2 completed in {task2_time/60:.1f} minutes")
        
        # Summary
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("🎉 COMPLETE PIPELINE SUCCESS")
        print("="*60)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📊 Repositories processed: {len(processed_df)}")
        print(f"🔧 Raw features (Task 1): {len(df.columns)}")
        print(f"⚡ Engineered features (Task 2): {len(processed_df.columns) - len(df.columns)}")
        print(f"📈 Total features: {len(processed_df.columns)}")
        print()
        print("📁 Final results:")
        print(f"   • Raw data: data/enriched_github_data.csv")
        print(f"   • Engineered features: data/bsv_prioritized_features.csv")  
        print(f"   • Feature metadata: data/bsv_prioritized_features_metadata.json")
        print()
        print("🚀 Ready for Task 3: LLM Pairwise Ranking")
        
        # Show top repositories by BSV score
        if 'bsv_investment_score' in processed_df.columns:
            print("\n📈 Top 10 Repositories by BSV Investment Score:")
            top_repos = processed_df.nlargest(10, 'bsv_investment_score')
            for i, (_, row) in enumerate(top_repos.iterrows(), 1):
                repo_name = row['repo_name'] if 'repo_name' in row else row['repo_url'].split('/')[-1]
                score = row['bsv_investment_score']
                funding_risk = row.get('funding_risk_level', 'unknown')
                print(f"   {i:2d}. {repo_name:<25} | Score: {score:.3f} | Funding: {funding_risk}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_complete_pipeline()