"""
Full Data Collection Runner
Executes complete Task 1 pipeline on all 100 BSV repositories
"""

import sys
import time
from pathlib import Path
from src.data_collection_runner import main as run_collection

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

if __name__ == "__main__":
    print("🚀 BSV Repository Prioritizer - Full Data Collection")
    print("=" * 60)
    print()
    print("This will collect comprehensive data for all 100 repositories")
    print("with all 4 Task 1 subtasks:")
    print("  ✅ 1.1 GitHub API Data Collection")  
    print("  ✅ 1.2 Code Quality Indicators")
    print("  ✅ 1.3 Adoption Signals")
    print("  ✅ 1.4 Funding Detection")
    print()
    print("Expected:")
    print(f"  📊 87 features per repository")
    print(f"  ⏱️  ~40 minutes total processing time")  
    print(f"  📁 Results saved to data/enriched_github_data.csv")
    print()
    
    # Check if user wants to proceed
    response = input("Proceed with full data collection? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("Starting full data collection...")
        start_time = time.time()
        
        try:
            run_collection()
            
            elapsed_time = time.time() - start_time
            print(f"\n🎉 Full data collection completed in {elapsed_time/60:.1f} minutes!")
            print("📊 Results available in data/enriched_github_data.csv")
            print("🚀 Ready to proceed to Task 2 (Feature Engineering)")
            
        except Exception as e:
            print(f"\n❌ Data collection failed: {e}")
            print("Please check your GitHub token and network connection")
            sys.exit(1)
    else:
        print("Full data collection cancelled.")
        print("To test the pipeline: python src/test_complete_task1.py")
        print("To proceed to Task 2: See Tasks/02_Feature_Engineering_and_Signals.md")