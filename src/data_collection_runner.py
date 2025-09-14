"""
Data Collection Runner
Orchestrates the collection of GitHub data from the provided dataset
"""

import pandas as pd
import os
import sys
from pathlib import Path
import logging
from github_collector import GitHubCollector

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the original dataset"""
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} repositories")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        raise

def extract_github_urls(df: pd.DataFrame) -> list:
    """Extract GitHub URLs from the dataset"""
    # The dataset has the first column as GitHub URLs
    urls = []
    for _, row in df.iterrows():
        url = str(row.iloc[0]).strip()  # First column contains the GitHub URLs
        if url.startswith('https://github.com/'):
            urls.append(url)
        else:
            logger.warning(f"Invalid GitHub URL format: {url}")
    
    logger.info(f"Extracted {len(urls)} valid GitHub URLs")
    return urls

def save_enriched_data(df: pd.DataFrame, output_path: str):
    """Save the enriched dataset"""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved enriched dataset to {output_path}")
        
        # Save summary statistics
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Data Collection Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total repositories processed: {len(df)}\n")
            f.write(f"Successfully collected: {len(df[df['error'].isna()])}\n")
            f.write(f"Failed collections: {len(df[~df['error'].isna()])}\n")
            f.write(f"Total features collected: {len(df.columns)}\n\n")
            
            f.write("Feature Summary:\n")
            for col in sorted(df.columns):
                non_null = df[col].notna().sum()
                f.write(f"  {col}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)\n")
        
        logger.info(f"Saved summary statistics to {summary_path}")
        
    except Exception as e:
        logger.error(f"Failed to save enriched dataset: {e}")
        raise

def main():
    """Main data collection pipeline"""
    # Paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "Docs" / "Dataset.csv"
    output_path = project_root / "data" / "enriched_github_data.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True)
    
    try:
        # Load original dataset
        logger.info("Starting data collection pipeline...")
        original_df = load_dataset(dataset_path)
        
        # Extract GitHub URLs
        github_urls = extract_github_urls(original_df)
        
        if not github_urls:
            logger.error("No valid GitHub URLs found in dataset")
            return
        
        # Initialize collector
        collector = GitHubCollector()
        
        # Collect comprehensive data
        logger.info("Beginning comprehensive data collection...")
        enriched_df = collector.collect_batch_data(github_urls)
        
        # Merge with original data
        # Create a mapping from URL to original data
        original_df['github_url'] = github_urls[:len(original_df)]
        merged_df = enriched_df.merge(
            original_df, 
            left_on='repo_url', 
            right_on='github_url', 
            how='left'
        )
        
        # Save results
        save_enriched_data(merged_df, output_path)
        
        # Print success summary
        successful_collections = len(merged_df[merged_df['error'].isna()])
        logger.info(f"Data collection completed! Successfully enriched {successful_collections}/{len(github_urls)} repositories")
        logger.info(f"Enriched dataset saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Data collection pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()