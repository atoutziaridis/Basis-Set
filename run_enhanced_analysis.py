#!/usr/bin/env python3
"""
Enhanced BSV Repository Prioritizer with Real OpenAI API Integration
Builds on existing implementation with:
1. Real OpenAI API calls for repository analysis
2. Enhanced technical execution scoring
3. Token-efficient batch processing
4. Improved market and team analysis
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import json
import logging
import re
import openai
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
import pickle
import gc
warnings.filterwarnings('ignore')

# Load environment variables from .env file (override existing ones)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Force override existing env vars
    print("✅ Loaded environment variables from .env file (with override)")
except ImportError:
    print("⚠️  python-dotenv not installed, trying to load .env manually")
    # Manual .env loading as fallback
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print("✅ Manually loaded environment variables from .env file")

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import existing components
from feature_engineer import FeatureEngineer
from src.final_scorer import FinalScorer

@dataclass
class LLMRepositoryAnalysis:
    """Structured LLM analysis result for a repository"""
    repository: str
    innovation_score: float  # 0-1
    technical_execution: float  # 0-1
    market_potential: float  # 0-1
    team_quality: float  # 0-1
    competitive_advantage: float  # 0-1
    overall_assessment: str
    key_strengths: List[str]
    key_risks: List[str]
    investment_thesis: str
    confidence: str  # low, medium, high
    tokens_used: int
    processing_time: float

class EnhancedLLMAnalyzer:
    """Enhanced LLM analyzer using real OpenAI API calls"""
    
    def __init__(self, model: str = "gpt-4o-mini", batch_size: int = 5, test_mode: bool = False):
        self.model = model  # Using gpt-4o-mini for cost efficiency
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.setup_logging()
        
        # Initialize OpenAI client only if not in test mode and API key is available
        if not test_mode:
            try:
                self.client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment
                self.logger.info("✅ OpenAI client initialized successfully")
            except Exception as e:
                self.logger.warning(f"⚠️  OpenAI client initialization failed: {e}")
                self.client = None
                self.test_mode = True
        else:
            self.client = None
            self.logger.info("🧪 Running in test mode - OpenAI calls disabled")
        
        # Token tracking
        self.total_tokens_used = 0
        self.total_cost_estimate = 0.0
        
        # Rate limiting
        self.requests_per_minute = 60  # Optimized rate limit
        self.last_request_time = 0
        self.api_response_times = []  # Track response times for dynamic adjustment
        self.timeout_seconds = 30  # Individual API call timeout
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('EnhancedLLMAnalyzer')
        
    async def analyze_repositories_batch(self, repositories: List[Dict]) -> List[LLMRepositoryAnalysis]:
        """Analyze repositories in efficient batches"""
        self.logger.info(f"🤖 Starting LLM analysis of {len(repositories)} repositories")
        self.logger.info(f"📊 Using model: {self.model}")
        self.logger.info(f"⚡ Batch size: {self.batch_size}")
        
        # If in test mode or no client, return fallback analyses
        if self.test_mode or self.client is None:
            self.logger.info("🧪 Test mode - generating fallback analyses")
            return [self._create_fallback_analysis(repo) for repo in repositories]
        
        all_results = []
        
        # Process in batches to manage rate limits and memory
        for i in range(0, len(repositories), self.batch_size):
            batch = repositories[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(repositories) + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} repositories)")
            
            # Process batch with rate limiting
            batch_results = await self._process_batch(batch)
            all_results.extend(batch_results)
            
            # Rate limiting between batches
            if i + self.batch_size < len(repositories):
                # Dynamic pause based on API performance
                avg_response_time = sum(self.api_response_times[-5:]) / min(5, len(self.api_response_times)) if self.api_response_times else 2
                pause_time = max(0.5, min(3.0, avg_response_time * 0.3))  # Adaptive pause
                await asyncio.sleep(pause_time)
                
        self.logger.info(f"✅ LLM analysis completed!")
        self.logger.info(f"📊 Total tokens used: {self.total_tokens_used:,}")
        self.logger.info(f"💰 Estimated cost: ${self.total_cost_estimate:.2f}")
        
        return all_results
        
    async def _process_batch(self, batch: List[Dict]) -> List[LLMRepositoryAnalysis]:
        """Process a batch of repositories"""
        tasks = []
        
        for repo_data in batch:
            task = self._analyze_single_repository(repo_data)
            tasks.append(task)
            
        # Execute batch with concurrency control
        semaphore = asyncio.Semaphore(8)  # Increased to 8 concurrent requests
        
        async def analyze_with_semaphore(task):
            async with semaphore:
                return await task
                
        results = await asyncio.gather(
            *[analyze_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Filter out exceptions and return successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, LLMRepositoryAnalysis):
                successful_results.append(result)
            else:
                self.logger.error(f"Analysis failed for {batch[i]['repository']}: {result}")
                # Create fallback analysis
                fallback = self._create_fallback_analysis(batch[i])
                successful_results.append(fallback)
                
        return successful_results
        
    async def _analyze_single_repository(self, repo_data: Dict) -> LLMRepositoryAnalysis:
        """Analyze a single repository with OpenAI API"""
        start_time = time.time()
        
        # Create token-efficient prompt
        prompt = self._create_efficient_prompt(repo_data)
        
        try:
            # Rate limiting
            await self._enforce_rate_limit()
            
            # Make API call
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistent analysis
                max_tokens=1500,  # More tokens for detailed analysis
                response_format={"type": "json_object"},
                timeout=self.timeout_seconds  # Add timeout to API call
            )
            
            # Track usage and response times
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            self.total_cost_estimate += self._calculate_cost(tokens_used)
            response_time = time.time() - start_time
            self.api_response_times.append(response_time)
            
            # Dynamic rate adjustment based on response times
            if len(self.api_response_times) > 10:
                avg_response_time = sum(self.api_response_times[-10:]) / 10
                if avg_response_time > 10:  # If responses are slow
                    self.requests_per_minute = max(30, self.requests_per_minute - 5)
                elif avg_response_time < 3:  # If responses are fast
                    self.requests_per_minute = min(90, self.requests_per_minute + 5)
            
            # Parse response
            analysis = self._parse_llm_response(
                response.choices[0].message.content,
                repo_data['repository'],
                tokens_used,
                time.time() - start_time
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed for {repo_data['repository']}: {e}")
            return self._create_fallback_analysis(repo_data, time.time() - start_time)
            
    def _create_efficient_prompt(self, repo_data: Dict) -> str:
        """Create comprehensive prompt for detailed repository analysis"""
        # Extract comprehensive metrics
        stars = repo_data.get('stars', 0)
        forks = repo_data.get('forks', 0)
        fork_ratio = forks / max(stars, 1)
        engagement = repo_data.get('engagement_score', 0)
        category_potential = repo_data.get('category_potential_score', 0)
        problem_ambition = repo_data.get('problem_ambition_score', 0)
        
        # Determine repository characteristics for focused analysis
        is_high_stars = stars > 10000
        is_enterprise_focus = 'enterprise' in repo_data.get('description', '').lower()
        is_dev_tool = any(word in repo_data.get('description', '').lower() 
                         for word in ['api', 'framework', 'library', 'tool', 'sdk'])
        
        # Create comprehensive analysis prompt
        prompt = f"""Conduct a detailed BSV investment analysis for this GitHub repository:

REPOSITORY PROFILE:
• Name: {repo_data['repository']}
• Stars: {stars:,} | Forks: {forks:,} | Fork Ratio: {fork_ratio:.3f}
• Language: {repo_data.get('language', 'Unknown')}
• Description: {repo_data.get('description', 'No description')[:300]}
• Website: {repo_data.get('website', 'None')}
• Engagement Score: {engagement:.3f}
• Category Potential: {category_potential:.3f}
• Problem Ambition: {problem_ambition:.3f}

CONTEXT ANALYSIS:
• High-visibility project: {'Yes' if is_high_stars else 'No'}
• Enterprise-focused: {'Yes' if is_enterprise_focus else 'No'}
• Developer tool/platform: {'Yes' if is_dev_tool else 'No'}

REQUIRED ANALYSIS (be specific and varied in your scores):

1. INNOVATION ASSESSMENT (0.0-1.0):
   - Technical novelty and breakthrough potential
   - Problem-solving approach uniqueness
   - Technology stack innovation

2. TECHNICAL EXECUTION (0.0-1.0):
   - Code architecture and quality indicators
   - Development practices and maintainability
   - Technical debt and scalability concerns

3. MARKET OPPORTUNITY (0.0-1.0):
   - Total addressable market size
   - Market timing and growth trajectory
   - Competitive landscape position

4. TEAM & COMMUNITY (0.0-1.0):
   - Community engagement and growth
   - Contributor quality and diversity
   - Long-term sustainability indicators

5. COMPETITIVE ADVANTAGE (0.0-1.0):
   - Differentiation from alternatives
   - Defensibility and network effects
   - Switching costs and moats

DETAILED OUTPUT REQUIRED:
• Numerical scores for each dimension (vary scores realistically)
• 3-sentence overall assessment
• Specific strengths (3-4 items with details)
• Key risks (2-3 items with mitigation strategies)
• Investment thesis (2 sentences with specific reasoning)
• Confidence level with justification

Format response as valid JSON with detailed, specific analysis."""
        
        return prompt
        
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM analysis"""
        return """You are an expert venture capital analyst specializing in early-stage technology investments. You evaluate GitHub repositories for their potential to become category-defining companies.

Focus on:
- Technical innovation and execution quality
- Market opportunity and timing
- Team capabilities and community strength
- Competitive differentiation
- Commercial viability

Be objective, analytical, and focus on investment potential over current popularity."""
        
    async def _enforce_rate_limit(self):
        """Enforce rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute  # seconds between requests
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
            
        self.last_request_time = time.time()
        
    def _parse_llm_response(self, response: str, repository: str, tokens_used: int, processing_time: float) -> LLMRepositoryAnalysis:
        """Parse LLM response into structured analysis"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            data = json.loads(json_str)
            
            return LLMRepositoryAnalysis(
                repository=repository,
                innovation_score=float(data.get('innovation', data.get('INNOVATION', 0.5))),
                technical_execution=float(data.get('execution', data.get('EXECUTION', 0.5))),
                market_potential=float(data.get('market', data.get('MARKET', 0.5))),
                team_quality=float(data.get('team', data.get('TEAM', 0.5))),
                competitive_advantage=float(data.get('advantage', data.get('ADVANTAGE', 0.5))),
                overall_assessment=data.get('assessment', data.get('overall_assessment', 'Analysis pending')),
                key_strengths=data.get('strengths', data.get('key_strengths', ['To be analyzed'])),
                key_risks=data.get('risks', data.get('key_risks', ['To be analyzed'])),
                investment_thesis=data.get('thesis', data.get('investment_thesis', 'Requires review')),
                confidence=data.get('confidence', 'medium'),
                tokens_used=tokens_used,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response for {repository}: {e}")
            return self._create_fallback_analysis({'repository': repository}, processing_time)
            
    def _create_fallback_analysis(self, repo_data: Dict, processing_time: float = 0.0) -> LLMRepositoryAnalysis:
        """Create fallback analysis when LLM fails"""
        return LLMRepositoryAnalysis(
            repository=repo_data['repository'],
            innovation_score=0.5,
            technical_execution=0.5,
            market_potential=0.5,
            team_quality=0.5,
            competitive_advantage=0.5,
            overall_assessment="Analysis unavailable due to processing error",
            key_strengths=["Unable to analyze"],
            key_risks=["Analysis incomplete"],
            investment_thesis="Requires manual review",
            confidence="low",
            tokens_used=0,
            processing_time=processing_time
        )
        
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate estimated cost for token usage"""
        # GPT-4o-mini pricing (as of 2024): $0.15/1M input tokens, $0.6/1M output tokens
        # Approximate 70% input, 30% output
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        input_cost = (input_tokens / 1_000_000) * 0.15
        output_cost = (output_tokens / 1_000_000) * 0.60
        
        return input_cost + output_cost

class EnhancedTechnicalAnalyzer:
    """Enhanced technical execution analyzer"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('EnhancedTechnicalAnalyzer')
        
    def calculate_enhanced_technical_score(self, repo_data: Dict, llm_analysis: Optional[LLMRepositoryAnalysis] = None) -> float:
        """Calculate comprehensive technical execution score using engineered features"""
        
        # Use engineered technical composite as primary signal (60%)
        base_technical = repo_data.get('technical_maturity_composite', 0.0)
        score = base_technical * 0.6
        
        # Enhance with LLM technical assessment if available (25%)
        if llm_analysis and llm_analysis.technical_execution:
            score += llm_analysis.technical_execution * 0.25
        else:
            # Fallback technical assessment
            score += self._assess_technical_fallback(repo_data) * 0.25
            
        # Add execution velocity composite (15%)
        execution_velocity = repo_data.get('execution_velocity_composite', 0.0)
        score += execution_velocity * 0.15
        
        return min(score, 1.0)
        
    def _assess_technical_fallback(self, repo_data: Dict) -> float:
        """Fallback technical assessment when LLM unavailable"""
        score = 0.0
        
        # Language sophistication
        lang_scores = {
            'Rust': 0.9, 'Go': 0.8, 'TypeScript': 0.7, 'Python': 0.6,
            'Java': 0.5, 'JavaScript': 0.4, 'C++': 0.8, 'C#': 0.5
        }
        score += lang_scores.get(repo_data.get('language', ''), 0.3)
        
        return min(score, 1.0)
        
    def _assess_code_quality(self, repo_data: Dict) -> float:
        """Assess code quality indicators"""
        score = 0.0
        
        # Language quality indicators
        lang = repo_data.get('language', '')
        if lang in ['Rust', 'Go']:
            score += 0.4  # Memory-safe, modern languages
        elif lang in ['TypeScript', 'Python']:
            score += 0.3  # Good tooling and practices
        elif lang in ['JavaScript', 'Java']:
            score += 0.2  # Established but variable quality
            
        # Repository maturity indicators
        stars = repo_data.get('stars', 0)
        if stars > 10000:
            score += 0.3  # Popular projects tend to have better quality
        elif stars > 1000:
            score += 0.2
        elif stars > 100:
            score += 0.1
            
        # Professional indicators
        if repo_data.get('website'):
            score += 0.1  # Professional presentation
            
        # Description quality (proxy for documentation)
        desc = repo_data.get('description', '')
        if desc and len(desc) > 100:
            score += 0.2
        elif desc and len(desc) > 50:
            score += 0.1
            
        return min(score, 1.0)
        
    def _assess_development_velocity(self, repo_data: Dict) -> float:
        """Assess development velocity and consistency"""
        score = 0.0
        
        # Activity indicators from engagement
        stars = repo_data.get('stars', 0)
        forks = repo_data.get('forks', 0)
        
        # High fork ratio indicates active development
        if forks > 0 and stars > 0:
            fork_ratio = forks / stars
            if 0.1 <= fork_ratio <= 0.3:  # Healthy development activity
                score += 0.4
            elif 0.05 <= fork_ratio < 0.1:
                score += 0.2
                
        # Community engagement suggests ongoing development
        if stars > 5000:
            score += 0.3  # Large projects require ongoing development
        elif stars > 1000:
            score += 0.2
            
        # Modern tech stack suggests active development
        lang = repo_data.get('language', '')
        if lang in ['Rust', 'Go', 'TypeScript']:
            score += 0.3  # Modern languages suggest recent development
        elif lang in ['Python', 'JavaScript']:
            score += 0.1
            
        return min(score, 1.0)
        
    def _assess_architecture(self, repo_data: Dict) -> float:
        """Assess architectural sophistication"""
        score = 0.0
        desc = str(repo_data.get('description', '')).lower()
        
        # Architecture keywords
        arch_indicators = {
            'microservice': 0.3, 'distributed': 0.3, 'scalable': 0.2,
            'api': 0.2, 'platform': 0.2, 'framework': 0.2,
            'infrastructure': 0.3, 'system': 0.2, 'engine': 0.2
        }
        
        for keyword, value in arch_indicators.items():
            if keyword in desc:
                score += value
                break  # Only count one primary architecture type
                
        # System-level languages suggest sophisticated architecture
        lang = repo_data.get('language', '')
        if lang in ['Rust', 'Go', 'C++']:
            score += 0.3  # Systems programming languages
        elif lang in ['Java', 'TypeScript']:
            score += 0.2  # Enterprise-ready languages
            
        # Scale indicators
        stars = repo_data.get('stars', 0)
        if stars > 20000:
            score += 0.4  # Large-scale projects need good architecture
        elif stars > 5000:
            score += 0.2
            
        return min(score, 1.0)

class EnhancedBSVAnalyzer:
    """Enhanced BSV analyzer with real OpenAI integration"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_logging()
        self.start_time = time.time()
        
        # Progress persistence
        self.progress_file = self.project_root / "data" / "analysis_progress.json"
        self.results_cache = self.project_root / "data" / "llm_results_cache.pkl"
        self.completed_repos = self._load_progress()
        
        # Initialize analyzers (check for OpenAI API key)
        api_key_available = os.getenv('OPENAI_API_KEY') is not None
        self.llm_analyzer = EnhancedLLMAnalyzer(test_mode=not api_key_available)
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        
        print("🚀 ENHANCED BSV REPOSITORY ANALYZER")
        print("=" * 60)
        print("🤖 Real OpenAI API Integration")
        print("⚡ Enhanced Technical Execution Scoring")
        print("💰 Token-Efficient Analysis")
        print("📊 Investment-Grade Results")
        print()
        
    def setup_logging(self):
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "enhanced_analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EnhancedBSVAnalyzer')
        
    def _format_list_field(self, field_value):
        """Format list field for CSV output, handling various data types"""
        if isinstance(field_value, list):
            # Handle list of strings or dicts
            formatted_items = []
            for item in field_value:
                if isinstance(item, dict):
                    # Convert dict to string representation
                    formatted_items.append(str(item))
                else:
                    formatted_items.append(str(item))
            return '; '.join(formatted_items)
        else:
            return str(field_value)
        
    async def run_enhanced_analysis(self, 
                                  features_csv_path: str = "data/final_engineered_features.csv",
                                  llm_rankings_path: str = "data/complete_real_dataset_llm_rankings.csv",
                                  existing_rankings_path: str = "output/bsv_complete_final_rankings.csv",
                                  resume_from_cache: bool = True) -> str:
        """Run enhanced analysis with real OpenAI API calls using all engineered features"""
        
        self.logger.info("🚀 Starting Enhanced BSV Analysis with Full Feature Integration")
        
        # Load comprehensive engineered features
        features_df = pd.read_csv(features_csv_path)
        self.logger.info(f"📊 Loaded engineered features: {len(features_df)} repositories, {len(features_df.columns)} features")
        
        # Load LLM rankings with real preference scores
        try:
            llm_rankings_df = pd.read_csv(llm_rankings_path)
            self.logger.info(f"📊 Loaded LLM rankings: {len(llm_rankings_df)} repositories with real preference scores")
        except:
            llm_rankings_df = None
            self.logger.info("📊 No LLM rankings found - will use fallback scoring")
        
        # Load existing rankings for comparison
        try:
            existing_df = pd.read_csv(existing_rankings_path)
            self.logger.info(f"📊 Loaded existing rankings for comparison: {len(existing_df)} repositories")
        except:
            existing_df = None
            self.logger.info("📊 No existing rankings found - proceeding with fresh analysis")
        
        # Convert to repository data format for LLM analysis
        repositories = []
        for _, row in features_df.iterrows():
            repo_data = {
                'repository': row['repository'],
                'stars': int(row.get('stars', 0)),
                'forks': int(row.get('forks', 0)),
                'description': str(row.get('description', '')),
                'language': str(row.get('language', '')),
                'website': str(row.get('website', '')),
                # Include engineered features for enhanced analysis
                'engagement_score': float(row.get('engagement_score', 0)),
                'execution_velocity_composite': float(row.get('execution_velocity_composite', 0)),
                'team_community_composite': float(row.get('team_community_composite', 0) if pd.notna(row.get('team_community_composite')) else 0),
                'technical_maturity_composite': float(row.get('technical_maturity_composite', 0)),
                'market_positioning_composite': float(row.get('market_positioning_composite', 0)),
                'bsv_investment_score': float(row.get('bsv_investment_score', 0)),
                'category_potential_score': float(row.get('category_potential_score', 0)),
            }
            repositories.append(repo_data)
            
        # Check for cached results and filter out completed repositories
        llm_results = []
        if resume_from_cache:
            cached_results = self._load_cached_results()
            if cached_results:
                self.logger.info(f"📚 Found {len(cached_results)} cached analyses")
                llm_results.extend(cached_results)
                completed_repo_names = {r.repository for r in cached_results}
                repositories = [r for r in repositories if r['repository'] not in completed_repo_names]
                self.logger.info(f"🔄 Resuming analysis for {len(repositories)} remaining repositories")
            
        # Step 1: Real LLM Analysis (with progress persistence)
        self.logger.info("📋 Step 1: Real OpenAI LLM Analysis")
        if repositories:  # Only analyze if there are uncached repositories
            new_results = await self.llm_analyzer.analyze_repositories_batch(repositories)
            llm_results.extend(new_results)
            # Save progress after batch completion
            completed_repo_names = {r.repository for r in new_results if r.tokens_used > 0}
            self.completed_repos.update(completed_repo_names)
            self._save_progress(self.completed_repos, new_results)
        else:
            self.logger.info("✅ All repositories already analyzed - using cached results")
            
        # Create lookup for LLM results
        llm_results_dict = {r.repository: r for r in llm_results}
        
        # Step 2: Enhanced Scoring Integration using FinalScorer
        self.logger.info("📋 Step 2: Integrating LLM Analysis with Engineered Features using FinalScorer")
        
        # Initialize FinalScorer with BSV-optimized weights
        final_scorer = FinalScorer()
        
        # Calculate final scores using the enhanced technical execution methods
        if llm_rankings_df is not None:
            final_results_df = final_scorer.calculate_final_scores(features_df, llm_rankings_df)
        else:
            # Create dummy LLM scores if not available
            dummy_llm_df = pd.DataFrame({
                'repository': features_df['repository'],
                'llm_preference_score': 0.5,
                'innovation_reasoning': 'No LLM analysis available',
                'innovation_category': 'moderate_innovation'
            })
            final_results_df = final_scorer.calculate_final_scores(features_df, dummy_llm_df)
        
        # Convert FinalScorer results to enhanced format for compatibility
        enhanced_data = []
        
        for _, scorer_row in final_results_df.iterrows():
            repo_name = scorer_row['repo_name']
            
            # Find corresponding repository data for additional fields
            repo_data = next((r for r in repositories if r['repository'] == repo_name), {})
            llm_analysis = llm_results_dict.get(repo_name, None)
            
            # Create comprehensive enhanced record using FinalScorer results
            enhanced_record = {
                'rank': scorer_row['rank'],
                'repository': repo_name,
                'final_score': scorer_row['final_score'],
                'llm_preference_score': scorer_row['llm_preference_score'],
                'technical_execution_score': scorer_row['technical_execution_score'], 
                'market_adoption_score': scorer_row['market_adoption_score'],
                'team_resilience_score': scorer_row['team_resilience_score'],
                'funding_gate_multiplier': scorer_row['funding_gate_multiplier'],
                'funding_risk_level': scorer_row.get('funding_risk_level', 'low_risk_unfunded'),
                
                # Basic repository info  
                'stars': scorer_row.get('stars', repo_data.get('stars', 0)),
                'forks': scorer_row.get('forks', repo_data.get('forks', 0)),
                'description': repo_data.get('description', ''),
                'language': repo_data.get('language', ''),
                'website': repo_data.get('website', ''),
                
                # Enhanced scores from FinalScorer
                'base_technical_composite': repo_data.get('technical_maturity_composite', 0.0),
                'base_market_composite': repo_data.get('market_positioning_composite', 0.0),
                'base_team_composite': repo_data.get('team_community_composite', 0.0),
                'engagement_score': repo_data.get('engagement_score', 0.0),
                'bsv_investment_score': scorer_row.get('bsv_investment_score', 1.0),
                'category_potential_score': scorer_row.get('category_potential_score', 1.0),
                
                # Metadata
                'methodology_version': '2.0_enhanced_openai_integrated',
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            }
            
            # Add LLM analysis details if available, otherwise use FinalScorer calculated values  
            if llm_analysis:
                enhanced_record.update({
                    'innovation_score': llm_analysis.innovation_score,
                    'competitive_advantage': llm_analysis.competitive_advantage,
                    'overall_assessment': llm_analysis.overall_assessment,
                    'key_strengths': self._format_list_field(llm_analysis.key_strengths),
                    'key_risks': self._format_list_field(llm_analysis.key_risks),
                    'investment_thesis': llm_analysis.investment_thesis,
                    'llm_confidence': llm_analysis.confidence,
                    'tokens_used': llm_analysis.tokens_used,
                })
            else:
                # Use FinalScorer calculated innovation and competitive advantage scores
                enhanced_record.update({
                    'innovation_score': scorer_row.get('innovation_score', 0.5),
                    'competitive_advantage': scorer_row.get('competitive_advantage', 0.5),
                    'overall_assessment': f'Enhanced analysis using {len(features_df.columns)} engineered features with BSV technical execution methods',
                    'key_strengths': 'Advanced technical execution scoring with AI/ML bias and time series analysis',
                    'key_risks': 'Limited LLM analysis available',
                    'investment_thesis': f'BSV Score: {scorer_row.get("bsv_investment_score", 0.0):.3f} | Technical: {scorer_row["technical_execution_score"]:.3f} | Market: {scorer_row["market_adoption_score"]:.3f}',
                    'llm_confidence': 'medium',
                    'tokens_used': 0,
                })
                
            enhanced_data.append(enhanced_record)
            
        # Create enhanced DataFrame
        enhanced_df = pd.DataFrame(enhanced_data)
        enhanced_df = enhanced_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        enhanced_df['rank'] = range(1, len(enhanced_df) + 1)
        
        # Save enhanced results
        output_path = self.project_root / "output" / "bsv_enhanced_openai_rankings.csv"
        enhanced_df.to_csv(output_path, index=False)
        
        # Generate summary
        total_time = time.time() - self.start_time
        total_tokens = sum(r.tokens_used for r in llm_results if r)
        total_cost = self.llm_analyzer.total_cost_estimate
        
        print()
        print("=" * 60)
        print("🎉 ENHANCED ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📊 Repositories analyzed: {len(enhanced_df)}")
        print(f"🤖 LLM analyses completed: {len([r for r in llm_results if r.tokens_used > 0])}")
        print(f"💰 Total tokens used: {total_tokens:,}")
        print(f"💸 Estimated cost: ${total_cost:.2f}")
        print(f"📁 Results saved to: {output_path}")
        print()
        
        # Show top 10 results with enhanced info
        print("🏆 TOP 10 ENHANCED RANKINGS:")
        print("-" * 80)
        for i, (_, row) in enumerate(enhanced_df.head(10).iterrows(), 1):
            repo_name = row['repository'].split('/')[-1]
            conf = row.get('llm_confidence', 'N/A')
            tokens = row.get('tokens_used', 0)
            status = "🤖 AI" if tokens > 0 else "📈 Eng"
            print(f"{i:2d}. {repo_name:<25} | {row['final_score']:.3f} | {conf} | {status} | ⭐{row['stars']:,}")
            
        # Summary statistics
        ai_analyzed = len([r for r in llm_results if r.tokens_used > 0])
        engineered_only = len(enhanced_df) - ai_analyzed
        print(f"\n📊 Analysis Summary:")
        print(f"  🤖 AI-Enhanced: {ai_analyzed} repositories")
        print(f"  📈 Feature-Based: {engineered_only} repositories")
        print(f"  🔄 Cache hit rate: {(engineered_only/len(enhanced_df)*100):.1f}%")
            
        return str(output_path)
        
    def _calculate_market_fallback(self, repo_data: Dict) -> float:
        """Fallback market scoring when LLM unavailable"""
        score = 0.0
        stars = repo_data.get('stars', 0)
        
        # Market traction based on stars
        if stars > 20000:
            score += 0.4
        elif stars > 5000:
            score += 0.3
        elif stars > 1000:
            score += 0.2
        elif stars > 100:
            score += 0.1
            
        # Market category assessment
        desc = str(repo_data.get('description', '')).lower()
        hot_markets = ['ai', 'ml', 'api', 'database', 'infrastructure', 'platform']
        if any(term in desc for term in hot_markets):
            score += 0.3
            
        # Developer ecosystem
        fork_ratio = repo_data.get('forks', 0) / max(repo_data.get('stars', 1), 1)
        if fork_ratio > 0.1:
            score += 0.3
        elif fork_ratio > 0.05:
            score += 0.2
            
        return min(score, 1.0)
        
    def _calculate_team_fallback(self, repo_data: Dict) -> float:
        """Fallback team scoring when LLM unavailable"""
        score = 0.0
        
        # Community engagement
        stars = repo_data.get('stars', 0)
        forks = repo_data.get('forks', 0)
        
        if forks > 100:
            score += 0.4  # Strong contributor base
        elif forks > 20:
            score += 0.3
        elif forks > 5:
            score += 0.2
            
        # Project management indicators
        if repo_data.get('website'):
            score += 0.3  # Professional presence
            
        # Community size
        if stars > 5000:
            score += 0.3
        elif stars > 1000:
            score += 0.2
            
        return min(score, 1.0)
        
    def _calculate_dynamic_weights(self, llm_analysis: Optional[LLMRepositoryAnalysis], 
                                 tech_score: float, market_score: float) -> Dict[str, float]:
        """Calculate dynamic weights based on analysis confidence and data quality"""
        
        # Base weights
        weights = {'llm': 0.50, 'technical': 0.20, 'market': 0.20, 'team': 0.10}
        
        # Adjust based on LLM confidence
        if llm_analysis:
            if llm_analysis.confidence == 'high':
                weights['llm'] += 0.05
                weights['technical'] -= 0.02
                weights['market'] -= 0.02
                weights['team'] -= 0.01
            elif llm_analysis.confidence == 'low':
                weights['llm'] -= 0.05
                weights['technical'] += 0.02
                weights['market'] += 0.02
                weights['team'] += 0.01
                
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
        
    def _load_progress(self) -> set:
        """Load previously completed repository analyses"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    return set(progress_data.get('completed_repos', []))
            except Exception as e:
                self.logger.warning(f"Failed to load progress: {e}")
        return set()
        
    def _save_progress(self, completed_repos: set, batch_results: List[LLMRepositoryAnalysis] = None):
        """Save progress and cache results"""
        try:
            # Save progress
            progress_data = {
                'completed_repos': list(completed_repos),
                'last_update': datetime.now().isoformat(),
                'total_tokens': self.llm_analyzer.total_tokens_used if hasattr(self, 'llm_analyzer') else 0
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
            # Cache batch results if provided
            if batch_results:
                existing_cache = []
                if self.results_cache.exists():
                    try:
                        with open(self.results_cache, 'rb') as f:
                            existing_cache = pickle.load(f)
                    except:
                        pass
                        
                existing_cache.extend(batch_results)
                with open(self.results_cache, 'wb') as f:
                    pickle.dump(existing_cache, f)
                    
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
            
    def _load_cached_results(self) -> List[LLMRepositoryAnalysis]:
        """Load cached LLM results"""
        if self.results_cache.exists():
            try:
                with open(self.results_cache, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cached results: {e}")
        return []
        
    def _stream_repositories_old(self, repositories: List[Dict], chunk_size: int = 20) -> List[List[Dict]]:
        """Legacy method - kept for compatibility"""
        chunks = []
        for i in range(0, len(repositories), chunk_size):
            chunk = repositories[i:i+chunk_size]
            chunks.append(chunk)
            # Trigger garbage collection between chunks
            if i > 0 and i % (chunk_size * 2) == 0:
                gc.collect()
        return chunks

# Main execution
async def main():
    """Run enhanced BSV analysis with optimizations"""
    analyzer = EnhancedBSVAnalyzer()
    
    try:
        result_path = await analyzer.run_enhanced_analysis(resume_from_cache=True)
        
        print(f"\n🚀 Enhanced Analysis Complete!")
        print(f"📊 Results: {result_path}")
        
        # Cleanup cache files if analysis is complete
        if analyzer.progress_file.exists():
            analyzer.progress_file.unlink()
        if analyzer.results_cache.exists():
            analyzer.results_cache.unlink()
        print("🧹 Cleaned up temporary cache files")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted - progress has been saved")
        print(f"💾 Resume later with: python {__file__}")
        return None
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("💾 Progress has been saved - you can resume the analysis")
        raise
        
    return result_path

if __name__ == "__main__":
    # Set event loop policy for better performance on some systems
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        result = asyncio.run(main())
        if result:
            print(f"\n✅ Success! Results saved to: {result}")
    except KeyboardInterrupt:
        print("\n👋 Analysis stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
