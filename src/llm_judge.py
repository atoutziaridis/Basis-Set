"""
LLM Judge Implementation - Task 3.3
Structured prompting with LLM APIs for pairwise repository comparison
"""

import openai
import json
import time
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import os
from dataclasses import dataclass
import re
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class JudgmentResult:
    """Structure for LLM judgment results"""
    winner: int  # 0 for repo1, 1 for repo2
    confidence: str  # 'low', 'medium', 'high'
    reasoning: str
    raw_response: str
    processing_time: float

class LLMJudge:
    """LLM-based pairwise repository judge"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", max_retries: int = 3):
        """Initialize LLM judge with API credentials"""
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass directly.")
        
        self.model = model
        self.max_retries = max_retries
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Cost and usage tracking
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        # Judgment cache to avoid re-running comparisons
        self.judgment_cache = {}
    
    def judge_pair(self, repo1_card: str, repo2_card: str, repo1_name: str, repo2_name: str) -> JudgmentResult:
        """Judge a single pair of repositories"""
        
        # Check cache first
        cache_key = self._create_cache_key(repo1_name, repo2_name)
        if cache_key in self.judgment_cache:
            logger.debug(f"Using cached result for {repo1_name} vs {repo2_name}")
            return self.judgment_cache[cache_key]
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Create prompt
                prompt = self._create_judgment_prompt(repo1_card, repo2_card, repo1_name, repo2_name)
                
                # Make API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # Deterministic responses
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )
                
                # Track usage
                self.total_tokens_used += response.usage.total_tokens
                self.total_requests += 1
                
                # Parse response
                result = self._parse_response(response.choices[0].message.content, 
                                            repo1_name, repo2_name, time.time() - start_time)
                
                # Cache result
                self.judgment_cache[cache_key] = result
                
                logger.debug(f"Judged {repo1_name} vs {repo2_name}: {['Repository 1', 'Repository 2'][result.winner]} wins ({result.confidence} confidence)")
                
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {repo1_name} vs {repo2_name}: {e}")
                if attempt == self.max_retries - 1:
                    self.failed_requests += 1
                    return self._create_fallback_result(repo1_name, repo2_name, str(e), time.time() - start_time)
                
                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for LLM judge"""
        
        return """You are an expert venture capital analyst evaluating GitHub repositories for their potential to become category-defining companies. Your task is to compare two repositories and determine which has higher potential for becoming a breakthrough company without current institutional funding.

EVALUATION CRITERIA (in order of importance):
1. **Market Ambition** (30%): Problem scope, addressable market size, potential for transformation
2. **Technical Execution** (25%): Code quality, development velocity, team capabilities
3. **Differentiation** (20%): Novel approach, technical innovation, competitive advantage
4. **Commercialization Potential** (15%): Business model clarity, adoption readiness, monetization path
5. **Team & Community** (10%): Team strength, community traction, growth trajectory

IMPORTANT GUIDELINES:
- Focus on POTENTIAL, not current popularity (don't be swayed by star counts alone)
- Prioritize INNOVATION and DIFFERENTIATION over incremental improvements
- Consider MARKET TIMING and OPPORTUNITY SIZE
- Look for signs of EXECUTION QUALITY and TECHNICAL SOPHISTICATION
- Downweight vanity metrics like stars if not backed by substance
- Consider COMMERCIAL VIABILITY and path to building a company

Respond ONLY with valid JSON in this exact format:
{
  "winner": 1 or 2,
  "confidence": "low" | "medium" | "high",
  "reasoning": "One sentence explanation focusing on key differentiators"
}"""
    
    def _create_judgment_prompt(self, repo1_card: str, repo2_card: str, repo1_name: str, repo2_name: str) -> str:
        """Create the judgment prompt for a specific pair"""
        
        return f"""Compare these two GitHub repositories for their potential to become category-defining companies:

**Repository 1: {repo1_name}**
{repo1_card}

**Repository 2: {repo2_name}**
{repo2_card}

Which repository has higher potential to become a category-defining company? Consider market ambition, technical execution, differentiation, commercialization potential, and team strength. Focus on breakthrough potential rather than current popularity metrics.

Respond with JSON indicating the winner (1 or 2), confidence level, and concise reasoning."""
    
    def _parse_response(self, response_text: str, repo1_name: str, repo2_name: str, processing_time: float) -> JudgmentResult:
        """Parse LLM response into structured result"""
        
        try:
            # Parse JSON response
            data = json.loads(response_text.strip())
            
            # Extract and validate fields
            winner = int(data.get('winner', 1)) - 1  # Convert to 0-based index
            if winner not in [0, 1]:
                raise ValueError(f"Invalid winner value: {winner + 1}")
            
            confidence = data.get('confidence', 'medium').lower()
            if confidence not in ['low', 'medium', 'high']:
                confidence = 'medium'
            
            reasoning = str(data.get('reasoning', 'No reasoning provided'))[:200]  # Limit length
            
            return JudgmentResult(
                winner=winner,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=response_text,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {response_text}")
            
            # Fallback parsing attempt
            return self._fallback_parse(response_text, repo1_name, repo2_name, processing_time)
    
    def _fallback_parse(self, response_text: str, repo1_name: str, repo2_name: str, processing_time: float) -> JudgmentResult:
        """Attempt to extract judgment from malformed response"""
        
        # Try to extract winner from text
        winner = 0  # Default to first repository
        confidence = 'low'  # Default to low confidence for fallback
        reasoning = "Parsing failed, using fallback analysis"
        
        # Look for winner indicators
        text_lower = response_text.lower()
        if any(term in text_lower for term in ['repository 2', 'repo 2', 'second', 'option 2']):
            winner = 1
        
        # Look for confidence indicators
        if any(term in text_lower for term in ['high confidence', 'strongly', 'clearly']):
            confidence = 'high'
        elif any(term in text_lower for term in ['medium', 'moderate']):
            confidence = 'medium'
        
        # Extract any reasoning
        if len(response_text) > 10:
            reasoning = response_text[:150] + "..." if len(response_text) > 150 else response_text
        
        return JudgmentResult(
            winner=winner,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=response_text,
            processing_time=processing_time
        )
    
    def _create_fallback_result(self, repo1_name: str, repo2_name: str, error_msg: str, processing_time: float) -> JudgmentResult:
        """Create fallback result when API call fails"""
        
        return JudgmentResult(
            winner=0,  # Default to first repository
            confidence='low',
            reasoning=f"API call failed: {error_msg[:100]}",
            raw_response=f"ERROR: {error_msg}",
            processing_time=processing_time
        )
    
    def _create_cache_key(self, repo1_name: str, repo2_name: str) -> str:
        """Create cache key for pair (order-independent)"""
        names = sorted([repo1_name, repo2_name])
        return f"{names[0]}_vs_{names[1]}"
    
    def judge_pairs_batch(self, pairs: List[Tuple[int, int]], cards: Dict[str, str], 
                         repo_names: List[str], batch_size: int = 10) -> List[JudgmentResult]:
        """Judge multiple pairs with batch processing and rate limiting"""
        
        logger.info(f"Starting batch judgment of {len(pairs)} pairs...")
        
        results = []
        
        for i, (idx1, idx2) in enumerate(pairs):
            try:
                # Get repository info
                repo1_name = repo_names[idx1] if idx1 < len(repo_names) else f"repo_{idx1}"
                repo2_name = repo_names[idx2] if idx2 < len(repo_names) else f"repo_{idx2}"
                
                repo1_card = cards.get(repo1_name, f"No card available for {repo1_name}")
                repo2_card = cards.get(repo2_name, f"No card available for {repo2_name}")
                
                # Judge pair
                result = self.judge_pair(repo1_card, repo2_card, repo1_name, repo2_name)
                results.append(result)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(pairs)} pairs...")
                
                # Rate limiting (respect API limits)
                if (i + 1) % batch_size == 0:
                    time.sleep(1)  # Brief pause between batches
                
            except Exception as e:
                logger.error(f"Failed to judge pair {i}: {e}")
                # Create error result
                error_result = self._create_fallback_result(
                    f"repo_{idx1}", f"repo_{idx2}", str(e), 0.0
                )
                results.append(error_result)
        
        logger.info(f"✅ Completed batch judgment: {len(results)} results")
        logger.info(f"📊 Usage: {self.total_requests} requests, {self.total_tokens_used} tokens, {self.failed_requests} failures")
        
        return results
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        
        estimated_cost = self._estimate_cost()
        
        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            'total_tokens': self.total_tokens_used,
            'estimated_cost_usd': estimated_cost,
            'cached_judgments': len(self.judgment_cache),
            'model_used': self.model
        }
    
    def _estimate_cost(self) -> float:
        """Estimate API cost based on token usage"""
        
        # OpenAI pricing (as of 2024, may change)
        pricing = {
            'gpt-3.5-turbo': 0.0015 / 1000,  # $0.0015 per 1K tokens
            'gpt-4': 0.03 / 1000,            # $0.03 per 1K tokens  
            'gpt-4-turbo': 0.01 / 1000       # $0.01 per 1K tokens
        }
        
        rate = pricing.get(self.model, 0.002 / 1000)  # Default fallback rate
        return self.total_tokens_used * rate
    
    def save_results(self, results: List[JudgmentResult], pairs: List[Tuple[int, int]], 
                    repo_names: List[str], output_path: str):
        """Save judgment results to file"""
        
        try:
            # Prepare results data
            results_data = []
            
            for i, (result, (idx1, idx2)) in enumerate(zip(results, pairs)):
                repo1_name = repo_names[idx1] if idx1 < len(repo_names) else f"repo_{idx1}"
                repo2_name = repo_names[idx2] if idx2 < len(repo_names) else f"repo_{idx2}"
                
                result_data = {
                    'pair_id': i,
                    'repo1_index': idx1,
                    'repo2_index': idx2,
                    'repo1_name': repo1_name,
                    'repo2_name': repo2_name,
                    'winner_index': idx1 if result.winner == 0 else idx2,
                    'winner_name': repo1_name if result.winner == 0 else repo2_name,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'processing_time': result.processing_time,
                    'raw_response': result.raw_response
                }
                
                results_data.append(result_data)
            
            # Complete output
            output_data = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'model_used': self.model,
                'total_comparisons': len(results),
                'usage_stats': self.get_usage_stats(),
                'results': results_data
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"✅ Saved {len(results)} judgment results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

if __name__ == "__main__":
    # Test the LLM judge
    print("🧠 Testing LLM Judge (requires OpenAI API key)")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  No OPENAI_API_KEY found. Set environment variable to test.")
        print("   export OPENAI_API_KEY=your_key_here")
        exit()
    
    try:
        # Initialize judge
        judge = LLMJudge(model="gpt-3.5-turbo")
        
        # Test with mock repository cards
        test_card_1 = """**pg_mooncake** (Mooncake-Labs)
Domain: Database
Description: Real-time analytics on Postgres tables

**Key Metrics:**
• Community: 1.7K stars, 46 forks, 14 contributors
• Age: 2 months (21 commits in 6mo, 3 releases)
• Tech: Rust, 631KB, MIT license

**Execution & Team:**
• Development Velocity: Limited (3 commits, 5 issues, 2 PRs in 30d)
• Team Health: Limited (12/26 active weeks, bus factor: 0.31)

**Technical Maturity:**
• Overall Quality: Basic (Tests, Docker)
• Documentation: Excellent README

**Market Position:**
• Market Potential: Basic (Early adoption stage)
• Growth: 136.2 stars/month, Unfunded"""
        
        test_card_2 = """**test-repo** (example-org)
Domain: Web Development
Description: Simple web framework for small projects

**Key Metrics:**
• Community: 50 stars, 5 forks, 2 contributors
• Age: 12 months (100 commits in 6mo, 1 releases)
• Tech: JavaScript, 200KB, MIT license

**Execution & Team:**
• Development Velocity: Fair (10 commits, 2 issues, 1 PRs in 30d)
• Team Health: Basic (8/26 active weeks, bus factor: 0.80)

**Technical Maturity:**
• Overall Quality: Fair (Tests)
• Documentation: Good README

**Market Position:**
• Market Potential: Fair (Some adoption)
• Growth: 4.2 stars/month, Unfunded"""
        
        # Test judgment
        print("Testing single judgment...")
        result = judge.judge_pair(test_card_1, test_card_2, "pg_mooncake", "test-repo")
        
        print(f"✅ Judgment complete:")
        print(f"   Winner: {'pg_mooncake' if result.winner == 0 else 'test-repo'}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Reasoning: {result.reasoning}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        
        # Show usage stats
        stats = judge.get_usage_stats()
        print(f"\n📊 Usage Stats:")
        print(f"   Requests: {stats['total_requests']}")
        print(f"   Tokens: {stats['total_tokens']}")
        print(f"   Estimated cost: ${stats['estimated_cost_usd']:.4f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure OpenAI API key is valid and you have sufficient credits.")