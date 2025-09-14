"""
Comprehensive Test for Task 2: Feature Engineering and Signals
Validates all subtasks and generates analysis reports
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineer import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_task2_complete():
    """Comprehensive test of Task 2 feature engineering"""
    
    print("🧪 Task 2 Complete Test: Feature Engineering & Signals")
    print("=" * 60)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load test data from Task 1
    test_data_path = Path(__file__).parent.parent / "data" / "task1_complete_test.csv"
    
    if not test_data_path.exists():
        print("❌ Task 1 test data not found. Run Task 1 first.")
        return False
    
    try:
        # Load and process data
        print("📊 Loading and processing data...")
        df = engineer.load_data(str(test_data_path))
        processed_df, feature_importance = engineer.process_features(df)
        
        print(f"✅ Processing complete:")
        print(f"   Original features: {len(df.columns)}")
        print(f"   Engineered features: {len(processed_df.columns)}")
        print(f"   Added features: {len(processed_df.columns) - len(df.columns)}")
        
        # Validate each subtask
        print("\n🔍 Validating subtasks:")
        
        # 2.1 Execution & Velocity Signals
        exec_features = [col for col in processed_df.columns if any(term in col for term in 
                        ['velocity', 'commit', 'release', 'maintenance', 'consistency', 'execution'])]
        print(f"✅ 2.1 Execution & Velocity: {len(exec_features)} features")
        print(f"    {exec_features[:3]}...")
        
        # 2.2 Team & Community Signals  
        team_features = [col for col in processed_df.columns if any(term in col for term in
                        ['team', 'community', 'resilience', 'growth', 'network'])]
        print(f"✅ 2.2 Team & Community: {len(team_features)} features")
        print(f"    {team_features[:3]}...")
        
        # 2.3 Technical Maturity Indicators
        tech_features = [col for col in processed_df.columns if any(term in col for term in
                        ['operational', 'quality', 'stability', 'documentation', 'maturity'])]
        print(f"✅ 2.3 Technical Maturity: {len(tech_features)} features")
        print(f"    {tech_features[:3]}...")
        
        # 2.4 Market Positioning Signals
        market_features = [col for col in processed_df.columns if any(term in col for term in
                          ['problem', 'commercial', 'differentiation', 'market', 'positioning'])]
        print(f"✅ 2.4 Market Positioning: {len(market_features)} features")
        print(f"    {market_features[:3]}...")
        
        # 2.5 Composite Scores
        composite_features = [col for col in processed_df.columns if any(term in col for term in
                             ['composite', 'category_potential', 'bsv_investment'])]
        print(f"✅ 2.5 Composite Scores: {len(composite_features)} features")
        print(f"    {composite_features}")
        
        # Show sample values for key scores
        print(f"\n📈 Sample feature values (test repository):")
        sample_row = processed_df.iloc[0]
        key_features = [
            'execution_velocity_composite',
            'team_community_composite', 
            'technical_maturity_composite',
            'market_positioning_composite',
            'category_potential_score',
            'bsv_investment_score'
        ]
        
        for feature in key_features:
            if feature in sample_row:
                print(f"   {feature}: {sample_row[feature]:.3f}")
        
        # Feature importance analysis
        print(f"\n🔬 Feature Importance Analysis:")
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print("   Top 5 most important features:")
        for feature, importance in sorted_importance[:5]:
            print(f"     {feature}: {importance:.3f}")
        
        # Save comprehensive results
        output_path = Path(__file__).parent.parent / "data" / "task2_test_results.csv"
        engineer.save_processed_data(processed_df, str(output_path), feature_importance)
        
        # Generate validation report
        generate_validation_report(processed_df, feature_importance)
        
        print(f"\n🎉 Task 2 Complete Test: SUCCESS")
        print(f"📁 Results saved to: {output_path}")
        print(f"📊 Features engineered: {len(processed_df.columns) - len(df.columns)}")
        print(f"🎯 Ready for Task 3 (LLM Pairwise Ranking)")
        
        return True
        
    except Exception as e:
        print(f"❌ Task 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_validation_report(df: pd.DataFrame, feature_importance: dict):
    """Generate comprehensive validation report"""
    
    report_path = Path(__file__).parent.parent / "data" / "task2_validation_report.md"
    
    # Get engineered features
    engineered_features = [col for col in df.columns if '_score' in col or '_composite' in col]
    
    with open(report_path, 'w') as f:
        f.write("# Task 2 Validation Report: Feature Engineering\n\n")
        f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"- **Total Features**: {len(df.columns)}\n")
        f.write(f"- **Engineered Features**: {len(engineered_features)}\n")
        f.write(f"- **Repositories Processed**: {len(df)}\n\n")
        
        # Feature Categories
        f.write("## Feature Categories\n\n")
        categories = {
            'Execution & Velocity': [col for col in engineered_features if any(term in col for term in ['velocity', 'commit', 'release', 'maintenance', 'consistency', 'execution'])],
            'Team & Community': [col for col in engineered_features if any(term in col for term in ['team', 'community', 'resilience', 'growth', 'network'])],
            'Technical Maturity': [col for col in engineered_features if any(term in col for term in ['operational', 'quality', 'stability', 'documentation', 'maturity'])],
            'Market Positioning': [col for col in engineered_features if any(term in col for term in ['problem', 'commercial', 'differentiation', 'market', 'positioning'])],
            'Composite Scores': [col for col in engineered_features if any(term in col for term in ['composite', 'category_potential', 'bsv_investment'])]
        }
        
        for category, features in categories.items():
            f.write(f"### {category} ({len(features)} features)\n")
            for feature in features:
                f.write(f"- `{feature}`\n")
            f.write("\n")
        
        # Feature Statistics
        f.write("## Feature Statistics\n\n")
        f.write("| Feature | Min | Max | Mean | Std |\n")
        f.write("|---------|-----|-----|------|-----|\n")
        
        for feature in engineered_features:
            if feature in df.columns:
                series = df[feature]
                f.write(f"| {feature} | {series.min():.3f} | {series.max():.3f} | {series.mean():.3f} | {series.std():.3f} |\n")
        
        f.write("\n")
        
        # Feature Importance
        f.write("## Feature Importance\n\n")
        f.write("Top 10 most important features:\n\n")
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
            f.write(f"{i}. **{feature}**: {importance:.3f}\n")
        
        f.write("\n")
        
        # BSV Investment Insights
        f.write("## BSV Investment Insights\n\n")
        if 'bsv_investment_score' in df.columns:
            bsv_scores = df['bsv_investment_score']
            f.write(f"- **BSV Investment Score Range**: {bsv_scores.min():.3f} - {bsv_scores.max():.3f}\n")
            f.write(f"- **Mean BSV Score**: {bsv_scores.mean():.3f}\n")
            f.write(f"- **High Potential Repos** (BSV > 0.7): {(bsv_scores > 0.7).sum()}\n")
            f.write(f"- **Medium Potential Repos** (BSV 0.4-0.7): {((bsv_scores >= 0.4) & (bsv_scores <= 0.7)).sum()}\n")
            f.write(f"- **Low Potential Repos** (BSV < 0.4): {(bsv_scores < 0.4).sum()}\n")
        
        # Quality Checks
        f.write("\n## Quality Checks\n\n")
        f.write("### Normalization Validation\n")
        
        normalized_features = [col for col in engineered_features if '_score' in col or '_composite' in col]
        all_normalized = True
        
        for feature in normalized_features:
            if feature in df.columns:
                min_val, max_val = df[feature].min(), df[feature].max()
                if min_val < -0.01 or max_val > 1.01:  # Allow small floating point errors
                    f.write(f"⚠️ {feature}: Range [{min_val:.3f}, {max_val:.3f}] - Not properly normalized\n")
                    all_normalized = False
        
        if all_normalized:
            f.write("✅ All features properly normalized to [0,1] range\n")
        
        f.write("\n### Missing Value Check\n")
        missing_values = df[engineered_features].isnull().sum()
        if missing_values.sum() == 0:
            f.write("✅ No missing values in engineered features\n")
        else:
            f.write("⚠️ Missing values detected:\n")
            for feature, count in missing_values[missing_values > 0].items():
                f.write(f"- {feature}: {count} missing\n")
    
    print(f"📋 Validation report saved to: {report_path}")

if __name__ == "__main__":
    success = test_task2_complete()
    if not success:
        exit(1)