"""
Quick test of Task 3 pipeline with simplified setup
"""

from src.task3_llm_ranking_pipeline import LLMRankingPipeline
import pandas as pd
from pathlib import Path

def quick_test():
    """Quick test with existing dataset"""
    
    print("🚀 Quick Task 3 Pipeline Test")
    
    # Use existing test dataset
    data_dir = Path(__file__).parent / "data"
    test_path = data_dir / "test_task3_dataset.csv"
    
    if not test_path.exists():
        print("❌ Test dataset not found")
        return False
    
    # Create pipeline with minimal comparisons
    pipeline = LLMRankingPipeline(
        target_comparisons=8,  # Very small for quick test
        openai_model="gpt-3.5-turbo"
    )
    
    try:
        print("🔄 Running pipeline...")
        final_rankings = pipeline.run_complete_pipeline(str(test_path))
        
        print("\n✅ Pipeline completed successfully!")
        print("\n🏆 Final Rankings:")
        print(final_rankings[['final_rank', 'repository', 'integrated_score']].to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Task 3 pipeline validation successful!")
    else:
        print("\n❌ Task 3 pipeline validation failed!")