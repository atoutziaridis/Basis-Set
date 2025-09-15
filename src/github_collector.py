"""
GitHub Data Collection Module
Collects comprehensive repository data using GitHub API
"""

import os
import time
import json
import pandas as pd
import requests
from github import Github
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
# Simple rate limiting implementation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubCollector:
    """Collects comprehensive GitHub repository data with rate limiting"""
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize collector with GitHub token"""
        self.token = github_token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN environment variable or pass token directly.")
        
        self.github = Github(self.token)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        })
        
        # Rate limiting: GitHub allows 5000 requests/hour for authenticated users
        self.rate_limit_calls = 0
        self.rate_limit_reset_time = time.time() + 3600
        
    def _make_api_request(self, url: str) -> Optional[Dict]:
        """Make rate-limited API request"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {url}: {e}")
            return None
    
    def extract_repo_owner_name(self, github_url: str) -> tuple:
        """Extract owner and repo name from GitHub URL"""
        if github_url.startswith('https://github.com/'):
            parts = github_url.replace('https://github.com/', '').strip('/').split('/')
            if len(parts) >= 2:
                return parts[0], parts[1]
        raise ValueError(f"Invalid GitHub URL format: {github_url}")
    
    def collect_repository_metadata(self, repo_url: str) -> Dict[str, Any]:
        """Collect basic repository metadata"""
        try:
            owner, repo_name = self.extract_repo_owner_name(repo_url)
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # Basic metadata
            metadata = {
                'repo_url': repo_url,
                'owner': owner,
                'repo_name': repo_name,
                'full_name': repo.full_name,
                'description': repo.description or '',
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'watchers': repo.watchers_count,
                'created_at': repo.created_at.isoformat() if repo.created_at else None,
                'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                'pushed_at': repo.pushed_at.isoformat() if repo.pushed_at else None,
                'language': repo.language,
                'size': repo.size,
                'default_branch': repo.default_branch,
                'archived': repo.archived,
                'disabled': repo.disabled,
                'private': repo.private,
                'fork': repo.fork,
                'has_issues': repo.has_issues,
                'has_projects': repo.has_projects,
                'has_wiki': repo.has_wiki,
                'has_downloads': repo.has_downloads,
                'license': repo.license.name if repo.license else None,
                'topics': json.dumps(list(repo.get_topics())),
                'open_issues_count': repo.open_issues_count,
            }
            
            # Language distribution
            try:
                languages = repo.get_languages()
                total_bytes = sum(languages.values())
                if total_bytes > 0:
                    language_percentages = {lang: (bytes_count / total_bytes) * 100 
                                          for lang, bytes_count in languages.items()}
                    metadata['primary_language'] = max(language_percentages.keys(), 
                                                     key=language_percentages.get)
                    metadata['language_diversity'] = len(languages)
                    metadata['languages_json'] = json.dumps(language_percentages)
                else:
                    metadata['primary_language'] = None
                    metadata['language_diversity'] = 0
                    metadata['languages_json'] = '{}'
            except Exception as e:
                logger.warning(f"Could not fetch languages for {repo_url}: {e}")
                metadata['primary_language'] = repo.language
                metadata['language_diversity'] = 1 if repo.language else 0
                metadata['languages_json'] = '{}'
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to collect metadata for {repo_url}: {e}")
            return {'repo_url': repo_url, 'error': str(e)}
    
    def collect_activity_metrics(self, repo_url: str) -> Dict[str, Any]:
        """Collect repository activity metrics"""
        try:
            owner, repo_name = self.extract_repo_owner_name(repo_url)
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # Activity metrics
            metrics = {}
            
            # Use GitHub Statistics API for efficient commit data collection
            try:
                # Get commit activity statistics (52 weeks of data)
                commit_activity = repo.get_stats_commit_activity()
                if commit_activity:
                    # Last 26 weeks (6 months) of commit activity
                    recent_weeks = commit_activity[-26:] if len(commit_activity) >= 26 else commit_activity
                    
                    # Calculate metrics from weekly data
                    total_commits_6m = sum(week.total for week in recent_weeks)
                    active_weeks_6m = sum(1 for week in recent_weeks if week.total > 0)
                    
                    metrics['commits_6_months'] = total_commits_6m
                    metrics['active_weeks_6_months'] = active_weeks_6m
                    metrics['avg_commits_per_week'] = total_commits_6m / 26 if total_commits_6m > 0 else 0
                    
                    # Recent activity (last 4 weeks approximation)
                    recent_4_weeks = recent_weeks[-4:] if len(recent_weeks) >= 4 else recent_weeks
                    metrics['commits_30_days'] = sum(week.total for week in recent_4_weeks)
                else:
                    # Fallback: use basic repository push date for activity estimation
                    metrics.update(self._estimate_activity_from_push_date(repo))
                    
            except Exception as e:
                logger.warning(f"Could not fetch commit statistics for {repo_url}: {e}")
                # Fallback: estimate from repository metadata
                metrics.update(self._estimate_activity_from_push_date(repo))
            
            # Release activity
            try:
                releases = list(repo.get_releases())
                metrics['total_releases'] = len(releases)
                
                if releases:
                    latest_release = releases[0]
                    metrics['latest_release_date'] = latest_release.created_at.isoformat()
                    metrics['days_since_last_release'] = (datetime.now() - latest_release.created_at.replace(tzinfo=None)).days
                    
                    # Release frequency
                    recent_releases = [r for r in releases if 
                                     (datetime.now() - r.created_at.replace(tzinfo=None)).days <= 365]
                    metrics['releases_last_year'] = len(recent_releases)
                else:
                    metrics['latest_release_date'] = None
                    metrics['days_since_last_release'] = None
                    metrics['releases_last_year'] = 0
                    
            except Exception as e:
                logger.warning(f"Could not fetch release data for {repo_url}: {e}")
                metrics['total_releases'] = 0
                metrics['latest_release_date'] = None
                metrics['days_since_last_release'] = None
                metrics['releases_last_year'] = 0
            
            # Issue and PR metrics
            try:
                # Recent issues  
                thirty_days_ago_tz = datetime.now().replace(tzinfo=None)
                thirty_days_ago = thirty_days_ago_tz - timedelta(days=30)
                recent_issues = list(repo.get_issues(state='all', since=thirty_days_ago))
                issues_only = [i for i in recent_issues if not i.pull_request]
                prs_only = [i for i in recent_issues if i.pull_request]
                
                metrics['issues_30_days'] = len(issues_only)
                metrics['prs_30_days'] = len(prs_only)
                
                # Calculate response times (sample from recent issues)
                response_times = []
                for issue in issues_only[:10]:  # Sample first 10
                    comments = list(issue.get_comments())
                    if comments:
                        first_response = min(comments, key=lambda c: c.created_at)
                        response_time = (first_response.created_at - issue.created_at).total_seconds() / 3600
                        response_times.append(response_time)
                
                if response_times:
                    metrics['avg_issue_response_time_hours'] = sum(response_times) / len(response_times)
                    metrics['median_issue_response_time_hours'] = sorted(response_times)[len(response_times)//2]
                else:
                    metrics['avg_issue_response_time_hours'] = None
                    metrics['median_issue_response_time_hours'] = None
                    
            except Exception as e:
                logger.warning(f"Could not fetch issue/PR data for {repo_url}: {e}")
                metrics['issues_30_days'] = 0
                metrics['prs_30_days'] = 0
                metrics['avg_issue_response_time_hours'] = None
                metrics['median_issue_response_time_hours'] = None
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect activity metrics for {repo_url}: {e}")
            return {'error': str(e)}
    
    def collect_contributor_data(self, repo_url: str) -> Dict[str, Any]:
        """Collect contributor information and bus factor"""
        try:
            owner, repo_name = self.extract_repo_owner_name(repo_url)
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            contributor_data = {}
            
            # Get contributors
            try:
                contributors = list(repo.get_contributors())
                contributor_data['total_contributors'] = len(contributors)
                
                if contributors:
                    # Contribution distribution for bus factor
                    contributions = [c.contributions for c in contributors]
                    total_contributions = sum(contributions)
                    
                    if total_contributions > 0:
                        max_contributions = max(contributions)
                        contributor_data['bus_factor'] = 1 - (max_contributions / total_contributions)
                        contributor_data['top_contributor_percentage'] = (max_contributions / total_contributions) * 100
                        
                        # Gini coefficient for contribution inequality
                        sorted_contribs = sorted(contributions)
                        n = len(sorted_contribs)
                        index = range(1, n + 1)
                        gini = 2 * sum(index[i] * sorted_contribs[i] for i in range(n)) / (n * sum(sorted_contribs)) - (n + 1) / n
                        contributor_data['contribution_gini'] = gini
                        
                        # Active contributors (top 80% of contributions)
                        sorted_contributors = sorted(contributors, key=lambda x: x.contributions, reverse=True)
                        cumulative_contributions = 0
                        active_contributors = 0
                        for contributor in sorted_contributors:
                            cumulative_contributions += contributor.contributions
                            active_contributors += 1
                            if cumulative_contributions >= 0.8 * total_contributions:
                                break
                        contributor_data['active_contributors'] = active_contributors
                    else:
                        contributor_data['bus_factor'] = 0
                        contributor_data['top_contributor_percentage'] = 100
                        contributor_data['contribution_gini'] = 0
                        contributor_data['active_contributors'] = 1
                else:
                    contributor_data['bus_factor'] = 0
                    contributor_data['top_contributor_percentage'] = 100
                    contributor_data['contribution_gini'] = 0
                    contributor_data['active_contributors'] = 0
                    
            except Exception as e:
                logger.warning(f"Could not fetch contributor data for {repo_url}: {e}")
                contributor_data['total_contributors'] = 0
                contributor_data['bus_factor'] = 0
                contributor_data['top_contributor_percentage'] = 100
                contributor_data['contribution_gini'] = 0
                contributor_data['active_contributors'] = 0
            
            return contributor_data
            
        except Exception as e:
            logger.error(f"Failed to collect contributor data for {repo_url}: {e}")
            return {'error': str(e)}
    
    def collect_code_quality_indicators(self, repo_url: str) -> Dict[str, Any]:
        """Collect code quality and development practice indicators"""
        try:
            owner, repo_name = self.extract_repo_owner_name(repo_url)
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            quality_data = {}
            
            # Check for CI/CD workflows
            try:
                workflows_path = ".github/workflows"
                workflows = []
                try:
                    contents = repo.get_contents(workflows_path)
                    if isinstance(contents, list):
                        workflows = [c.name for c in contents if c.name.endswith(('.yml', '.yaml'))]
                    quality_data['has_ci_cd'] = len(workflows) > 0
                    quality_data['workflow_count'] = len(workflows)
                    quality_data['workflow_files'] = workflows[:5]  # Limit to first 5
                except:
                    quality_data['has_ci_cd'] = False
                    quality_data['workflow_count'] = 0
                    quality_data['workflow_files'] = []
                    
            except Exception as e:
                logger.warning(f"Could not check CI/CD for {repo_url}: {e}")
                quality_data['has_ci_cd'] = False
                quality_data['workflow_count'] = 0
                quality_data['workflow_files'] = []
            
            # Check for tests
            try:
                test_indicators = ['test', 'tests', 'spec', 'specs', '__tests__']
                has_tests = False
                test_directories = []
                
                def check_directory_for_tests(path=""):
                    nonlocal has_tests, test_directories
                    try:
                        contents = repo.get_contents(path)
                        if isinstance(contents, list):
                            for content in contents[:20]:  # Limit to avoid rate limits
                                if content.type == "dir":
                                    dir_name = content.name.lower()
                                    if any(indicator in dir_name for indicator in test_indicators):
                                        has_tests = True
                                        test_directories.append(content.name)
                                elif content.type == "file":
                                    file_name = content.name.lower()
                                    if any(f"{indicator}." in file_name or f".{indicator}." in file_name 
                                          for indicator in test_indicators):
                                        has_tests = True
                    except:
                        pass
                
                check_directory_for_tests()  # Check root
                
                quality_data['has_tests'] = has_tests
                quality_data['test_directories'] = test_directories
                
            except Exception as e:
                logger.warning(f"Could not check tests for {repo_url}: {e}")
                quality_data['has_tests'] = False
                quality_data['test_directories'] = []
            
            # Check for documentation
            try:
                # Check for README
                readme_content = ""
                readme_score = 0
                
                try:
                    readme = repo.get_readme()
                    readme_content = readme.decoded_content.decode('utf-8')
                    readme_score = self._calculate_readme_quality_score(readme_content)
                except:
                    pass
                
                quality_data['readme_length'] = len(readme_content)
                quality_data['readme_quality_score'] = readme_score
                
                # Check for docs directory
                has_docs = False
                try:
                    repo.get_contents("docs")
                    has_docs = True
                except:
                    try:
                        repo.get_contents("doc")
                        has_docs = True
                    except:
                        pass
                
                quality_data['has_docs_directory'] = has_docs
                
            except Exception as e:
                logger.warning(f"Could not check documentation for {repo_url}: {e}")
                quality_data['readme_length'] = 0
                quality_data['readme_quality_score'] = 0
                quality_data['has_docs_directory'] = False
            
            # Check for configuration files
            try:
                config_files = {
                    'dockerfile': ['Dockerfile', 'dockerfile'],
                    'docker_compose': ['docker-compose.yml', 'docker-compose.yaml', 'compose.yml'],
                    'package_json': ['package.json'],
                    'requirements_txt': ['requirements.txt'],
                    'pyproject_toml': ['pyproject.toml'],
                    'cargo_toml': ['Cargo.toml'],
                    'makefile': ['Makefile', 'makefile'],
                    'eslintrc': ['.eslintrc', '.eslintrc.js', '.eslintrc.json'],
                    'prettier': ['.prettierrc', 'prettier.config.js'],
                    'gitignore': ['.gitignore']
                }
                
                config_presence = {}
                for config_type, filenames in config_files.items():
                    found = False
                    for filename in filenames:
                        try:
                            repo.get_contents(filename)
                            found = True
                            break
                        except:
                            continue
                    config_presence[f'has_{config_type}'] = found
                
                quality_data.update(config_presence)
                
                # Calculate overall configuration score
                config_score = sum(config_presence.values()) / len(config_presence)
                quality_data['config_completeness_score'] = config_score
                
            except Exception as e:
                logger.warning(f"Could not check config files for {repo_url}: {e}")
                # Set defaults
                for config_type in config_files.keys():
                    quality_data[f'has_{config_type}'] = False
                quality_data['config_completeness_score'] = 0.0
            
            return quality_data
            
        except Exception as e:
            logger.error(f"Failed to collect code quality indicators for {repo_url}: {e}")
            return {'error': str(e)}
    
    def _calculate_readme_quality_score(self, readme_content: str) -> float:
        """Calculate README quality score based on various indicators"""
        score = 0.0
        content = readme_content.lower()
        
        # Length check (reasonable length gets points)
        if 500 <= len(readme_content) <= 10000:
            score += 0.2
        elif len(readme_content) > 10000:
            score += 0.1
        
        # Check for key sections
        sections = {
            'installation': ['install', 'setup', 'getting started'],
            'usage': ['usage', 'example', 'how to use'],
            'api': ['api', 'reference', 'documentation'],
            'contributing': ['contribut', 'development', 'pull request'],
            'license': ['license', 'licence']
        }
        
        for section, keywords in sections.items():
            if any(keyword in content for keyword in keywords):
                score += 0.15
        
        # Check for code examples
        if '```' in readme_content or '    ' in readme_content:  # Code blocks
            score += 0.1
        
        # Check for images/badges
        if '![' in readme_content or 'badge' in content:
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def collect_adoption_signals(self, repo_url: str) -> Dict[str, Any]:
        """Collect adoption and community signals"""
        try:
            owner, repo_name = self.extract_repo_owner_name(repo_url)
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            adoption_data = {}
            
            # Basic community metrics (already collected, but calculate ratios)
            stars = repo.stargazers_count
            forks = repo.forks_count
            
            adoption_data['fork_to_star_ratio'] = forks / max(stars, 1)
            adoption_data['stars_per_month'] = self._calculate_stars_growth_rate(repo)
            
            # Try to get package download statistics
            adoption_data.update(self._get_package_downloads(owner, repo_name, repo.language))
            
            # Get dependents information
            dependents_info = self._scrape_dependents_info(repo_url)
            adoption_data.update(dependents_info)
            
            # Network effect metrics
            try:
                # Get network info
                network_count = repo.network_count if hasattr(repo, 'network_count') else forks
                adoption_data['network_count'] = network_count
                
                # Subscribers (watching)
                adoption_data['subscribers_count'] = repo.subscribers_count
                
                # Calculate engagement score
                total_activity = stars + forks + repo.subscribers_count + repo.open_issues_count
                adoption_data['engagement_score'] = total_activity / max(repo.size, 1) if repo.size > 0 else total_activity
                
            except Exception as e:
                logger.warning(f"Could not get network metrics for {repo_url}: {e}")
                adoption_data['network_count'] = forks
                adoption_data['subscribers_count'] = 0
                adoption_data['engagement_score'] = 0
            
            return adoption_data
            
        except Exception as e:
            logger.error(f"Failed to collect adoption signals for {repo_url}: {e}")
            return {'error': str(e)}
    
    def _calculate_stars_growth_rate(self, repo) -> float:
        """Calculate approximate stars growth rate per month"""
        try:
            if repo.created_at and repo.stargazers_count > 0:
                age_months = (datetime.now() - repo.created_at.replace(tzinfo=None)).days / 30.44
                if age_months > 0:
                    return repo.stargazers_count / age_months
            return 0.0
        except:
            return 0.0
    
    def _get_package_downloads(self, owner: str, repo_name: str, language: str) -> Dict[str, Any]:
        """Get package download statistics with improved package name detection"""
        download_data = {
            'pypi_downloads': 0,
            'npm_downloads': 0,
            'cargo_downloads': 0,
            'has_package': False,
            'package_names_found': []
        }
        
        try:
            # Generate potential package names
            package_candidates = self._generate_package_candidates(owner, repo_name)
            
            # Check PyPI for Python projects
            if language and language.lower() in ['python']:
                for candidate in package_candidates:
                    pypi_data = self._check_pypi_downloads(candidate)
                    if pypi_data['has_package']:
                        download_data.update(pypi_data)
                        download_data['package_names_found'].append(f"pypi:{candidate}")
                        break  # Use first successful match
            
            # Check npm for JavaScript/TypeScript projects
            if language and language.lower() in ['javascript', 'typescript', 'vue', 'react']:
                for candidate in package_candidates:
                    npm_data = self._check_npm_downloads(candidate)
                    if npm_data['has_package']:
                        download_data.update(npm_data)
                        download_data['package_names_found'].append(f"npm:{candidate}")
                        break  # Use first successful match
            
            # Check crates.io for Rust projects
            if language and language.lower() in ['rust']:
                for candidate in package_candidates:
                    cargo_data = self._check_cargo_downloads(candidate)
                    if cargo_data['has_package']:
                        download_data.update(cargo_data)
                        download_data['package_names_found'].append(f"cargo:{candidate}")
                        break  # Use first successful match
                        
        except Exception as e:
            logger.warning(f"Could not get package downloads for {owner}/{repo_name}: {e}")
        
        return download_data
    
    def _estimate_activity_from_push_date(self, repo) -> Dict[str, int]:
        """Estimate activity metrics from repository push date when statistics unavailable"""
        try:
            now = datetime.now()
            
            # Check if repository was recently updated
            if repo.pushed_at:
                days_since_push = (now - repo.pushed_at.replace(tzinfo=None)).days
                
                # Estimate activity based on recency and repository popularity
                stars = repo.stargazers_count or 0
                forks = repo.forks_count or 0
                
                # Popular repositories tend to have more activity
                popularity_factor = min(stars + forks, 1000) / 1000
                
                # Recent updates suggest ongoing development
                if days_since_push <= 30:
                    recency_factor = 1.0
                elif days_since_push <= 90:
                    recency_factor = 0.7
                elif days_since_push <= 180:
                    recency_factor = 0.4
                else:
                    recency_factor = 0.1
                
                # Estimate commit metrics
                estimated_commits_6m = int(popularity_factor * recency_factor * 50)  # Conservative estimate
                estimated_active_weeks = int(min(26, estimated_commits_6m / 2)) if estimated_commits_6m > 0 else 0
                estimated_avg_commits = estimated_commits_6m / 26 if estimated_commits_6m > 0 else 0
                estimated_commits_30d = int(estimated_commits_6m * 0.3) if days_since_push <= 30 else 0
                
                return {
                    'commits_6_months': estimated_commits_6m,
                    'active_weeks_6_months': estimated_active_weeks,
                    'avg_commits_per_week': estimated_avg_commits,
                    'commits_30_days': estimated_commits_30d
                }
            
        except Exception as e:
            logger.warning(f"Error estimating activity from push date: {e}")
        
        # Final fallback - minimal activity
        return {
            'commits_6_months': 0,
            'active_weeks_6_months': 0,
            'avg_commits_per_week': 0,
            'commits_30_days': 0
        }
    
    def _generate_package_candidates(self, owner: str, repo_name: str) -> List[str]:
        """Generate potential package names based on common naming patterns"""
        candidates = []
        
        # Original repo name
        candidates.append(repo_name)
        
        # Common variations
        candidates.append(repo_name.lower())
        candidates.append(repo_name.replace('-', '_'))  # Python style
        candidates.append(repo_name.replace('_', '-'))  # npm style
        candidates.append(repo_name.replace('_', '').replace('-', ''))  # no separators
        
        # Owner prefix variations
        candidates.append(f"{owner}-{repo_name}")
        candidates.append(f"{owner}_{repo_name}")
        candidates.append(f"{owner}.{repo_name}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen and candidate:
                seen.add(candidate)
                unique_candidates.append(candidate)
                
        return unique_candidates[:5]  # Limit to top 5 candidates to avoid spam
    
    def _check_pypi_downloads(self, package_name: str) -> Dict[str, Any]:
        """Check PyPI download statistics with improved error handling"""
        for attempt in range(2):  # Retry once
            try:
                # First try PyPI API for package existence
                pypi_url = f"https://pypi.org/pypi/{package_name}/json"
                response = requests.get(pypi_url, timeout=10)
                
                if response.status_code == 200:
                    # Package exists, now get download stats
                    stats_url = f"https://pypistats.org/api/packages/{package_name}/recent"
                    stats_response = requests.get(stats_url, timeout=10)
                    
                    downloads = 0
                    if stats_response.status_code == 200:
                        stats_data = stats_response.json()
                        downloads = stats_data.get('data', {}).get('last_month', 0)
                    
                    return {
                        'pypi_downloads': downloads, 
                        'has_package': True
                    }
                elif response.status_code == 404:
                    # Package definitely doesn't exist
                    return {'pypi_downloads': 0, 'has_package': False}
                    
            except requests.exceptions.Timeout:
                if attempt == 0:
                    time.sleep(1)  # Brief pause before retry
                    continue
            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                    continue
                logger.debug(f"PyPI check failed for {package_name}: {e}")
                
        return {'pypi_downloads': 0, 'has_package': False}
    
    def _check_npm_downloads(self, package_name: str) -> Dict[str, Any]:
        """Check npm download statistics with improved error handling"""
        for attempt in range(2):  # Retry once
            try:
                # First check if package exists
                registry_url = f"https://registry.npmjs.org/{package_name}"
                response = requests.get(registry_url, timeout=10)
                
                if response.status_code == 200:
                    # Package exists, get download stats
                    downloads_url = f"https://api.npmjs.org/downloads/point/last-month/{package_name}"
                    downloads_response = requests.get(downloads_url, timeout=10)
                    
                    downloads = 0
                    if downloads_response.status_code == 200:
                        downloads_data = downloads_response.json()
                        downloads = downloads_data.get('downloads', 0)
                    
                    return {
                        'npm_downloads': downloads,
                        'has_package': True
                    }
                elif response.status_code == 404:
                    # Package doesn't exist
                    return {'npm_downloads': 0, 'has_package': False}
                    
            except requests.exceptions.Timeout:
                if attempt == 0:
                    time.sleep(1)  # Brief pause before retry
                    continue
            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                    continue
                logger.debug(f"npm check failed for {package_name}: {e}")
                
        return {'npm_downloads': 0, 'has_package': False}
    
    def _check_cargo_downloads(self, package_name: str) -> Dict[str, Any]:
        """Check crates.io download statistics with improved error handling"""
        for attempt in range(2):  # Retry once
            try:
                # crates.io API endpoint
                url = f"https://crates.io/api/v1/crates/{package_name}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    crate_info = data.get('crate', {})
                    downloads = crate_info.get('downloads', 0)
                    recent_downloads = crate_info.get('recent_downloads', downloads)
                    
                    return {
                        'cargo_downloads': recent_downloads,
                        'has_package': True
                    }
                elif response.status_code == 404:
                    # Crate doesn't exist
                    return {'cargo_downloads': 0, 'has_package': False}
                    
            except requests.exceptions.Timeout:
                if attempt == 0:
                    time.sleep(1)  # Brief pause before retry
                    continue
            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                    continue
                logger.debug(f"Cargo check failed for {package_name}: {e}")
                
        return {'cargo_downloads': 0, 'has_package': False}
    
    def _scrape_dependents_info(self, repo_url: str) -> Dict[str, Any]:
        """Scrape GitHub dependents information (lightweight)"""
        dependents_data = {
            'dependents_count': 0,
            'dependents_scraped': False
        }
        
        try:
            # Construct dependents URL
            dependents_url = f"{repo_url}/network/dependents"
            
            response = requests.get(dependents_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for dependents count in various possible locations
                count_elements = soup.find_all(text=re.compile(r'\d+\s+(repositories|repository)'))
                if count_elements:
                    for element in count_elements:
                        match = re.search(r'(\d+)', element)
                        if match:
                            dependents_data['dependents_count'] = int(match.group(1))
                            dependents_data['dependents_scraped'] = True
                            break
                
        except Exception as e:
            logger.warning(f"Could not scrape dependents for {repo_url}: {e}")
        
        return dependents_data
    
    def collect_funding_detection(self, repo_url: str) -> Dict[str, Any]:
        """Detect potential institutional funding through text analysis"""
        try:
            owner, repo_name = self.extract_repo_owner_name(repo_url)
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            funding_data = {}
            
            # Collect text sources for analysis
            text_sources = []
            
            # 1. Repository README
            readme_text = ""
            try:
                readme = repo.get_readme()
                readme_text = readme.decoded_content.decode('utf-8').lower()
                text_sources.append(('readme', readme_text))
            except:
                pass
            
            # 2. Repository description
            if repo.description:
                desc_text = repo.description.lower()
                text_sources.append(('description', desc_text))
            
            # 3. Organization information
            org_text = ""
            try:
                if repo.organization:
                    org = repo.organization
                    org_info = f"{org.name or ''} {org.bio or ''} {org.blog or ''} {org.location or ''}"
                    org_text = org_info.lower()
                    text_sources.append(('organization', org_text))
            except:
                pass
            
            # 4. Owner information
            try:
                owner_obj = repo.owner
                owner_info = f"{owner_obj.name or ''} {owner_obj.bio or ''} {owner_obj.blog or ''} {owner_obj.location or ''}"
                owner_text = owner_info.lower()
                text_sources.append(('owner', owner_text))
            except:
                pass
            
            # 5. Website content (if available and safe to scrape)
            website_text = ""
            if repo.homepage:
                website_text = self._safe_website_scrape(repo.homepage)
                if website_text:
                    text_sources.append(('website', website_text))
            
            # Analyze all text sources for funding indicators
            funding_analysis = self._analyze_funding_indicators(text_sources)
            funding_data.update(funding_analysis)
            
            # Generate overall funding assessment
            funding_data['funding_confidence'] = self._calculate_funding_confidence(funding_analysis)
            funding_data['funding_risk_level'] = self._categorize_funding_risk(funding_analysis)
            funding_data['text_sources_analyzed'] = len(text_sources)
            
            return funding_data
            
        except Exception as e:
            logger.error(f"Failed to collect funding detection for {repo_url}: {e}")
            return {'error': str(e)}
    
    def _safe_website_scrape(self, url: str) -> str:
        """Safely scrape website content with restrictions"""
        try:
            # Only scrape if it looks like a reasonable URL
            if not url.startswith(('http://', 'https://')):
                return ""
            
            # Avoid scraping large platforms or suspicious URLs
            blocked_domains = ['github.com', 'twitter.com', 'facebook.com', 'linkedin.com']
            if any(domain in url.lower() for domain in blocked_domains):
                return ""
            
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200 and len(response.content) < 500000:  # Limit to 500KB
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text from key sections
                text_content = ""
                for tag in soup.find_all(['title', 'h1', 'h2', 'h3', 'p'], limit=50):
                    text_content += f" {tag.get_text()}"
                return text_content.lower()[:2000]  # Limit to 2000 chars
        except Exception as e:
            logger.warning(f"Could not scrape website {url}: {e}")
        
        return ""
    
    def _analyze_funding_indicators(self, text_sources: List[tuple]) -> Dict[str, Any]:
        """Analyze text for funding-related keywords and patterns"""
        
        # Define funding indicator patterns
        funding_patterns = {
            'direct_funding': [
                r'series [a-z]', r'seed round', r'pre-seed', r'series a', r'series b', 
                r'series c', r'funding round', r'raised \$', r'investment of', 
                r'funding of', r'venture funding', r'equity funding'
            ],
            'investor_mentions': [
                r'backed by', r'supported by', r'invested by', r'funded by',
                r'venture capital', r'vc fund', r'angel investor', r'investment firm',
                r'venture partner', r'lead investor'
            ],
            'financial_terms': [
                r'\$\d+[mk]', r'\$\d+\s*million', r'\$\d+\s*billion', r'valuation of',
                r'worth \$', r'valued at', r'market cap', r'unicorn', r'decacorn'
            ],
            'company_stage': [
                r'startup', r'early stage', r'growth stage', r'scale-up',
                r'funded startup', r'venture-backed', r'portfolio company'
            ]
        }
        
        analysis = {}
        total_matches = 0
        matches_by_source = {}
        
        # Analyze each text source
        for source_name, text in text_sources:
            source_matches = {}
            source_total = 0
            
            for category, patterns in funding_patterns.items():
                matches = []
                for pattern in patterns:
                    found = re.findall(pattern, text)
                    matches.extend(found)
                
                source_matches[f'{category}_count'] = len(matches)
                source_matches[f'{category}_matches'] = matches[:3]  # Limit to first 3
                source_total += len(matches)
            
            matches_by_source[source_name] = source_matches
            matches_by_source[f'{source_name}_total'] = source_total
            total_matches += source_total
        
        # Aggregate results
        for category in funding_patterns.keys():
            category_total = sum(matches_by_source.get(source, {}).get(f'{category}_count', 0) 
                               for source, _ in text_sources)
            analysis[f'total_{category}_indicators'] = category_total
        
        analysis['total_funding_indicators'] = total_matches
        analysis['funding_indicators_by_source'] = matches_by_source
        
        # Special checks for strong positive/negative indicators
        all_text = " ".join([text for _, text in text_sources])
        
        # Strong negative indicators (clearly unfunded)
        negative_patterns = [
            r'no funding', r'bootstrapped', r'self-funded', r'independent',
            r'side project', r'personal project', r'hobby project'
        ]
        negative_matches = sum(len(re.findall(pattern, all_text)) for pattern in negative_patterns)
        analysis['negative_funding_indicators'] = negative_matches
        
        # Strong positive indicators (clearly funded)
        positive_patterns = [
            r'series [a-z] funding', r'raised \$\d+', r'venture capital',
            r'unicorn', r'ipo', r'acquisition'
        ]
        positive_matches = sum(len(re.findall(pattern, all_text)) for pattern in positive_patterns)
        analysis['strong_positive_indicators'] = positive_matches
        
        return analysis
    
    def _calculate_funding_confidence(self, funding_analysis: Dict) -> float:
        """Calculate confidence score for funding status (0-1 scale)"""
        
        # Weight different types of indicators
        direct_weight = 0.4
        investor_weight = 0.3
        financial_weight = 0.2
        stage_weight = 0.1
        
        # Get indicator counts
        direct = funding_analysis.get('total_direct_funding_indicators', 0)
        investor = funding_analysis.get('total_investor_mentions_indicators', 0)
        financial = funding_analysis.get('total_financial_terms_indicators', 0)
        stage = funding_analysis.get('total_company_stage_indicators', 0)
        
        # Calculate weighted score
        raw_score = (
            direct * direct_weight +
            investor * investor_weight +
            financial * financial_weight +
            stage * stage_weight
        )
        
        # Apply strong indicator adjustments
        strong_positive = funding_analysis.get('strong_positive_indicators', 0)
        negative = funding_analysis.get('negative_funding_indicators', 0)
        
        # Boost for strong positive indicators
        if strong_positive > 0:
            raw_score += strong_positive * 0.3
        
        # Reduce for negative indicators
        if negative > 0:
            raw_score -= negative * 0.2
        
        # Normalize to 0-1 scale with sigmoid-like function
        confidence = min(1.0, max(0.0, raw_score / 5.0))
        
        return round(confidence, 3)
    
    def _categorize_funding_risk(self, funding_analysis: Dict) -> str:
        """Categorize funding risk level for BSV filtering"""
        
        confidence = self._calculate_funding_confidence(funding_analysis)
        strong_positive = funding_analysis.get('strong_positive_indicators', 0)
        negative = funding_analysis.get('negative_funding_indicators', 0)
        
        if strong_positive > 0 or confidence > 0.7:
            return "high_risk_funded"
        elif confidence > 0.4:
            return "medium_risk_funded"  
        elif negative > 0 or confidence < 0.1:
            return "low_risk_unfunded"
        else:
            return "uncertain_funding_status"
    
    def collect_comprehensive_data(self, repo_url: str) -> Dict[str, Any]:
        """Collect all data for a single repository"""
        logger.info(f"Collecting data for {repo_url}")
        
        data = {}
        
        # Collect all data types
        metadata = self.collect_repository_metadata(repo_url)
        data.update(metadata)
        
        if 'error' not in metadata:
            activity = self.collect_activity_metrics(repo_url)
            data.update(activity)
            
            contributors = self.collect_contributor_data(repo_url)
            data.update(contributors)
            
            # New subtasks: Code Quality, Adoption Signals, and Funding Detection
            quality = self.collect_code_quality_indicators(repo_url)
            data.update(quality)
            
            adoption = self.collect_adoption_signals(repo_url)
            data.update(adoption)
            
            funding = self.collect_funding_detection(repo_url)
            data.update(funding)
        
        # Add collection timestamp
        data['collected_at'] = datetime.now().isoformat()
        
        return data
    
    def collect_batch_data(self, repo_urls: List[str]) -> pd.DataFrame:
        """Collect data for multiple repositories"""
        logger.info(f"Starting data collection for {len(repo_urls)} repositories")
        
        all_data = []
        
        for repo_url in tqdm(repo_urls, desc="Collecting repository data"):
            try:
                repo_data = self.collect_comprehensive_data(repo_url)
                all_data.append(repo_data)
                
                # Add small delay to be respectful to API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to collect data for {repo_url}: {e}")
                all_data.append({
                    'repo_url': repo_url,
                    'error': str(e),
                    'collected_at': datetime.now().isoformat()
                })
        
        return pd.DataFrame(all_data)

if __name__ == "__main__":
    # Test with a few repositories
    test_repos = [
        "https://github.com/microsoft/vscode",
        "https://github.com/python/cpython",
        "https://github.com/tensorflow/tensorflow"
    ]
    
    collector = GitHubCollector()
    df = collector.collect_batch_data(test_repos)
    print(df.head())
    print(f"Collected {len(df.columns)} features for {len(df)} repositories")