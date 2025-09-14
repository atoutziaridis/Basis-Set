"""
Repository Card Generator - Task 3.1
Generates structured summaries of repositories for LLM comparison
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RepositoryCardGenerator:
    """Generates structured repository cards for LLM pairwise comparison"""
    
    def __init__(self, token_target: int = 450):
        """Initialize with target token count"""
        self.token_target = token_target
        self.template_sections = {
            'header': 80,      # Name, domain, description
            'metrics': 120,    # Key numbers and stats
            'execution': 100,  # Development activity
            'technical': 80,   # Code quality and maturity
            'market': 70       # Positioning and viability
        }
        
    def generate_card(self, repo_data: pd.Series) -> str:
        """Generate a structured repository card from feature data"""
        
        try:
            # Header section
            header = self._generate_header(repo_data)
            
            # Key metrics section
            metrics = self._generate_metrics(repo_data)
            
            # Execution section
            execution = self._generate_execution(repo_data)
            
            # Technical maturity section
            technical = self._generate_technical(repo_data)
            
            # Market positioning section
            market = self._generate_market(repo_data)
            
            # Combine all sections
            card = f"{header}\n\n{metrics}\n\n{execution}\n\n{technical}\n\n{market}"
            
            return card.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate card for {repo_data.get('repo_name', 'unknown')}: {e}")
            return self._generate_fallback_card(repo_data)
    
    def _generate_header(self, data: pd.Series) -> str:
        """Generate header section with name, domain, and description"""
        
        repo_name = data.get('repo_name', 'Unknown')
        owner = data.get('owner', 'Unknown')
        description = str(data.get('description', ''))
        
        # Clean and truncate description
        if description and description != 'nan':
            description = description.strip()
            if len(description) > 150:
                description = description[:147] + "..."
        else:
            description = "No description provided"
        
        # Infer domain from topics and language
        domain = self._infer_domain(data)
        
        header = f"**{repo_name}** ({owner})\n"
        header += f"Domain: {domain}\n"
        header += f"Description: {description}"
        
        return header
    
    def _infer_domain(self, data: pd.Series) -> str:
        """Infer project domain from topics, language, and description"""
        
        topics_str = str(data.get('topics', '')).lower()
        language = str(data.get('primary_language', '')).lower()
        description = str(data.get('description', '')).lower()
        
        combined_text = f"{topics_str} {description}"
        
        # Domain mapping based on keywords
        domains = {
            'AI/ML': ['machine-learning', 'ai', 'neural', 'deep-learning', 'tensorflow', 'pytorch'],
            'Database': ['database', 'sql', 'postgres', 'mongodb', 'analytics', 'data'],
            'DevOps': ['devops', 'kubernetes', 'docker', 'ci-cd', 'deployment', 'infrastructure'],
            'Web Framework': ['web', 'framework', 'api', 'server', 'http', 'rest'],
            'Developer Tools': ['cli', 'tool', 'utility', 'development', 'build', 'testing'],
            'Security': ['security', 'auth', 'encryption', 'vulnerability', 'pentesting'],
            'Blockchain': ['blockchain', 'crypto', 'bitcoin', 'ethereum', 'smart-contract'],
            'Mobile': ['mobile', 'android', 'ios', 'react-native', 'flutter'],
            'System/Low-level': ['system', 'kernel', 'compiler', 'runtime', 'performance']
        }
        
        # Check for domain matches
        for domain, keywords in domains.items():
            if any(keyword in combined_text for keyword in keywords):
                return domain
        
        # Fallback to language-based domain
        language_domains = {
            'rust': 'System/Low-level',
            'go': 'Backend/Infrastructure', 
            'python': 'Data/Automation',
            'javascript': 'Web Development',
            'typescript': 'Web Development',
            'java': 'Enterprise Software',
            'c++': 'System/Performance',
            'c': 'System/Low-level'
        }
        
        return language_domains.get(language, 'Software Tool')
    
    def _generate_metrics(self, data: pd.Series) -> str:
        """Generate key metrics section"""
        
        stars = self._format_number(data.get('stars', 0))
        forks = self._format_number(data.get('forks', 0))
        contributors = self._format_number(data.get('total_contributors', 0))
        
        # Repository age
        created_at = data.get('created_at')
        age_months = self._calculate_age_months(created_at)
        
        # Activity indicators
        commits_6m = data.get('commits_6_months', 0)
        releases = data.get('total_releases', 0)
        
        metrics = f"**Key Metrics:**\n"
        metrics += f"• Community: {stars} stars, {forks} forks, {contributors} contributors\n"
        metrics += f"• Age: {age_months} months ({commits_6m} commits in 6mo, {releases} releases)\n"
        
        # Language and size
        language = data.get('primary_language', 'Unknown')
        size_kb = data.get('size', 0)
        metrics += f"• Tech: {language}, {self._format_size(size_kb)}, "
        
        # License
        license_name = str(data.get('license', '')).replace(' License', '')
        metrics += f"{license_name} license"
        
        return metrics
    
    def _generate_execution(self, data: pd.Series) -> str:
        """Generate execution and velocity section"""
        
        # Get composite scores (normalized 0-1)
        exec_score = data.get('execution_velocity_composite', 0)
        team_score = data.get('team_community_composite', 0)
        
        # Recent activity
        commits_30d = data.get('commits_30_days', 0)
        issues_30d = data.get('issues_30_days', 0)
        prs_30d = data.get('prs_30_days', 0)
        
        # Development patterns
        active_weeks = data.get('active_weeks_6_months', 0)
        
        execution = f"**Execution & Team:**\n"
        execution += f"• Development Velocity: {self._score_to_rating(exec_score)} "
        execution += f"({commits_30d} commits, {issues_30d} issues, {prs_30d} PRs in 30d)\n"
        execution += f"• Team Health: {self._score_to_rating(team_score)} "
        execution += f"({active_weeks}/26 active weeks, bus factor: {data.get('bus_factor', 0):.2f})"
        
        return execution
    
    def _generate_technical(self, data: pd.Series) -> str:
        """Generate technical maturity section"""
        
        tech_score = data.get('technical_maturity_composite', 0)
        
        # Quality indicators
        has_ci_cd = data.get('has_ci_cd', False)
        has_tests = data.get('has_tests', False)
        has_docker = data.get('has_dockerfile', False)
        readme_quality = data.get('readme_quality_score', 0)
        
        # Build quality indicators list
        quality_items = []
        if has_ci_cd:
            quality_items.append("CI/CD")
        if has_tests:
            quality_items.append("Tests")
        if has_docker:
            quality_items.append("Docker")
        
        if quality_items:
            quality_text = ", ".join(quality_items)
        else:
            quality_text = "Basic setup"
            
        technical = f"**Technical Maturity:**\n"
        technical += f"• Overall Quality: {self._score_to_rating(tech_score)} "
        technical += f"({quality_text})\n"
        technical += f"• Documentation: {self._score_to_rating(readme_quality)} README"
        
        # Add package/distribution info
        has_package = data.get('has_package', False)
        if has_package:
            technical += f", Published package"
        
        return technical
    
    def _generate_market(self, data: pd.Series) -> str:
        """Generate market positioning section"""
        
        market_score = data.get('market_positioning_composite', 0)
        
        # Adoption indicators
        dependents = data.get('dependents_count', 0)
        downloads = (data.get('pypi_downloads', 0) + 
                    data.get('npm_downloads', 0) + 
                    data.get('cargo_downloads', 0))
        
        # Growth and positioning
        stars_per_month = data.get('stars_per_month', 0)
        funding_risk = data.get('funding_risk_level', 'unknown')
        
        market = f"**Market Position:**\n"
        market += f"• Market Potential: {self._score_to_rating(market_score)} "
        
        if dependents > 0 or downloads > 0:
            if dependents > 0:
                market += f"({dependents} dependents"
                if downloads > 0:
                    market += f", {self._format_number(downloads)} downloads"
                market += ")"
            else:
                market += f"({self._format_number(downloads)} downloads)"
        else:
            market += "(Early adoption stage)"
            
        market += f"\n• Growth: {stars_per_month:.1f} stars/month, {self._format_funding_status(funding_risk)}"
        
        return market
    
    def _score_to_rating(self, score: float) -> str:
        """Convert 0-1 score to human readable rating"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Basic"
        else:
            return "Limited"
    
    def _format_funding_status(self, funding_risk: str) -> str:
        """Format funding status for readability"""
        status_map = {
            'low_risk_unfunded': 'Unfunded',
            'uncertain_funding_status': 'Funding unclear',
            'medium_risk_funded': 'Likely funded',
            'high_risk_funded': 'Well funded'
        }
        return status_map.get(funding_risk, 'Unknown funding')
    
    def _format_number(self, num: int) -> str:
        """Format number with K/M suffixes"""
        if num >= 1000000:
            return f"{num/1000000:.1f}M"
        elif num >= 1000:
            return f"{num/1000:.1f}K"
        else:
            return str(int(num))
    
    def _format_size(self, size_kb: int) -> str:
        """Format repository size"""
        if size_kb >= 1000:
            return f"{size_kb/1000:.1f}MB"
        else:
            return f"{size_kb}KB"
    
    def _calculate_age_months(self, created_at_str: str) -> int:
        """Calculate repository age in months"""
        try:
            if pd.isna(created_at_str):
                return 0
            
            created_date = pd.to_datetime(created_at_str, utc=True)
            now = pd.Timestamp.now(tz='UTC')
            age_days = (now - created_date).days
            return max(int(age_days / 30.44), 1)
        except Exception:
            return 1
    
    def _generate_fallback_card(self, data: pd.Series) -> str:
        """Generate minimal fallback card when main generation fails"""
        repo_name = data.get('repo_name', 'Unknown Repository')
        description = str(data.get('description', 'No description available'))[:100]
        
        return f"**{repo_name}**\nDescription: {description}\nStatus: Limited data available"
    
    def generate_cards_batch(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate cards for all repositories in DataFrame"""
        
        logger.info(f"Generating cards for {len(df)} repositories...")
        
        cards = {}
        for idx, row in df.iterrows():
            repo_name = row.get('repo_name', f"repo_{idx}")
            card = self.generate_card(row)
            cards[repo_name] = card
            
        logger.info(f"✅ Generated {len(cards)} repository cards")
        return cards
    
    def save_cards(self, cards: Dict[str, str], output_path: str):
        """Save generated cards to file"""
        
        try:
            # Save as JSON for structured access
            cards_data = {
                'generated_at': datetime.now().isoformat(),
                'total_cards': len(cards),
                'cards': cards
            }
            
            with open(output_path, 'w') as f:
                json.dump(cards_data, f, indent=2)
            
            logger.info(f"✅ Saved {len(cards)} cards to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cards: {e}")
            raise

if __name__ == "__main__":
    # Test the card generator
    test_data_path = Path(__file__).parent.parent / "data" / "task2_engineered_features.csv"
    
    if test_data_path.exists():
        generator = RepositoryCardGenerator()
        
        # Load test data
        df = pd.read_csv(test_data_path)
        
        # Generate cards
        cards = generator.generate_cards_batch(df)
        
        # Save results
        output_path = Path(__file__).parent.parent / "data" / "repository_cards.json"
        generator.save_cards(cards, str(output_path))
        
        # Show sample card
        if cards:
            sample_name = list(cards.keys())[0]
            sample_card = cards[sample_name]
            print(f"📋 Sample Repository Card for {sample_name}:")
            print("=" * 50)
            print(sample_card)
            print("=" * 50)
            print(f"✅ Generated cards for {len(cards)} repositories")
    else:
        print("❌ Task 2 engineered features not found. Run Task 2 first.")