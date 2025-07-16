import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from pathlib import Path
import json
import yfinance as yf
import torch
from transformers import AutoTokenizer, AutoModel

from .classes import FeaturizedSample


class BaseFeaturizer(ABC):
    """Abstract base class for feature extractors"""
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names in consistent order"""
        pass
    
    @abstractmethod
    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features and return as dictionary"""
        pass

class SubmissionFeaturizer:
    """
    Main featurizer class that combines multiple feature extraction strategies
    """
    
    def __init__(self, feature_extractors: Optional[List[BaseFeaturizer]] = None):
        """
        Initialize with list of feature extractors
        
        Args:
            feature_extractors: List of feature extractor classes
        """
        if feature_extractors is None:
            # Default feature extractors
            self.feature_extractors = [
                TickerFeaturizer(),
                KeywordFeaturizer(),
                PsychologyFeaturizer(),
                StructureFeaturizer(),
                EngagementFeaturizer(),
                TextQualityFeaturizer(),
                WeeklyDiscussionFeaturizer(),
                SlidingWindowEmbeddingFeaturizer()
            ]
        else:
            self.feature_extractors = feature_extractors
        
        # Build consolidated feature names
        self.feature_names = []
        for extractor in self.feature_extractors:
            self.feature_names.extend(extractor.get_feature_names())
    
    def extract_features(self, submission_data: Dict[str, Any]) -> FeaturizedSample:
        """
        Extract all features from submission
        
        Args:
            submission_data: JSON submission data
            
        Returns:
            FeaturizedSample with all extracted features
        """
        submission_id = submission_data.get('submission_id', 'unknown')
        subreddit = submission_data.get('subreddit', 'unknown')
        date = submission_data.get('date', 'unknown')
        
        # Extract features from all extractors
        all_features = {}
        extractor_metadata = {}
        
        for extractor in self.feature_extractors:
            try:
                extractor_features = extractor.extract_features(submission_data)
                all_features.update(extractor_features)
                extractor_metadata[extractor.__class__.__name__] = {
                    'feature_count': len(extractor_features),
                    'success': True
                }
            except Exception as e:
                # Handle extractor failures gracefully
                extractor_name = extractor.__class__.__name__
                extractor_metadata[extractor_name] = {
                    'feature_count': 0,
                    'success': False,
                    'error': str(e)
                }
                # Fill with zeros for failed extractor
                for feature_name in extractor.get_feature_names():
                    all_features[feature_name] = 0.0
        
        # Create feature vector in consistent order
        feature_vector = np.array([all_features.get(name, 0.0) for name in self.feature_names])
        
        return FeaturizedSample(
            submission_id=submission_id,
            subreddit=subreddit,
            date=date,
            features=all_features,
            feature_vector=feature_vector,
            feature_names=self.feature_names.copy(),
            metadata={
                'extractors': extractor_metadata,
                'total_features': len(self.feature_names),
                'submission_title': submission_data.get('title', ''),
                'submission_subreddit': submission_data.get('subreddit', '')
            }
        )
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names in consistent order"""
        return self.feature_names.copy()
    
    def extract_batch(self, submissions: List[Dict[str, Any]]) -> List[FeaturizedSample]:
        """Extract features from multiple submissions"""
        return [self.extract_features(sub) for sub in submissions]

# =============================================================================
# Individual Feature Extractors
# =============================================================================

class TickerFeaturizer(BaseFeaturizer):
    """Extract ticker-related features"""
    
    def __init__(self, ticker_file: Optional[Path] = None):
        self.known_tickers = self._load_tickers_yfinance(ticker_file)
        self.ticker_patterns = [
            r'\$[A-Z]{1,5}\b',           # $AAPL
            r'\b[A-Z]{1,5}(?=\s|$)',     # AAPL (standalone)
            r'\b[A-Z]{2,5}(?=\s+stock)', # AAPL stock
        ]
    
    def get_feature_names(self) -> List[str]:
        return [
            'ticker_count',
            'explicit_ticker_count',
            'ticker_in_title',
            'ticker_density',
            'unique_tickers',
            'ticker_title_ratio'
        ]
    
    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        title = submission_data.get('title', '')
        comments = submission_data.get('comments', [])
        all_text = f"{title} " + " ".join(comments)
        
        # Find all tickers
        all_tickers = set()
        explicit_tickers = set()
        title_tickers = set()
        
        for pattern in self.ticker_patterns:
            # All text
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                clean_ticker = match.replace('$', '').upper()
                if clean_ticker in self.known_tickers:
                    all_tickers.add(clean_ticker)
                    if '$' in match:
                        explicit_tickers.add(clean_ticker)
            
            # Title only
            title_matches = re.findall(pattern, title, re.IGNORECASE)
            for match in title_matches:
                clean_ticker = match.replace('$', '').upper()
                if clean_ticker in self.known_tickers:
                    title_tickers.add(clean_ticker)
        
        # Calculate features
        word_count = len(all_text.split())
        title_word_count = len(title.split())
        
        return {
            'ticker_count': float(len(all_tickers)),
            'explicit_ticker_count': float(len(explicit_tickers)),
            'ticker_in_title': float(len(title_tickers) > 0),
            'ticker_density': (len(all_tickers) / max(word_count, 1)) * 100,
            'unique_tickers': float(len(all_tickers)),
            'ticker_title_ratio': len(title_tickers) / max(title_word_count, 1)
        }
    
    def _load_tickers(self, ticker_file: Optional[Path]) -> set:
        """Load ticker symbols from file or use defaults"""
        if ticker_file and ticker_file.exists():
            with open(ticker_file, 'r') as f:
                return set(line.strip().upper() for line in f)
        
        # Default tickers
        return {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.A', 'BRK.B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'AVGO',
            'KO', 'MRK', 'COST', 'PEP', 'TMO', 'WMT', 'ACN', 'MCD', 'ABT', 'VZ', 'ADBE',
            'NFLX', 'CRM', 'NKE', 'DHR', 'TXN', 'NEE', 'RTX', 'QCOM', 'ORCL', 'CMCSA',
            'AMD', 'INTC', 'IBM', 'UBER', 'PYPL', 'SHOP', 'SQ', 'ROKU', 'ZM', 'PTON'
        }

    def _load_tickers_yfinance(self, ticker_file: Optional[Path]) -> set:
        """Load ticker symbols using yfinance"""
        if ticker_file and ticker_file.exists():
            with open(ticker_file, 'r') as f:
                return set(line.strip().upper() for line in f)
        
        try:
            # Get S&P 500 tickers from Wikipedia via yfinance
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = set(sp500['Symbol'].str.replace('.', '-').tolist())  # yfinance uses - instead of .
            
            # Add some common additional tickers
            additional_tickers = {
                'BRK-A', 'BRK-B',  # Berkshire Hathaway
                'GOOGL', 'GOOG',   # Alphabet classes
            }
            tickers.update(additional_tickers)
            
            print(f"Loaded {len(tickers)} tickers from S&P 500")
            return tickers
            
        except Exception as e:
            print(f"Failed to load tickers from yfinance: {e}")
            return self._load_tickers(ticker_file)
        

class WeeklyDiscussionFeaturizer(BaseFeaturizer):
    """Extract weekly discussion features"""
    
    def get_feature_names(self) -> List[str]:
        return [
            'is_penny_stock_discussion',
            'is_earnings_discussion',
            'is_theta_daily_discussion',
            'is_daily_stock_discussion',
            'is_daily_discussion_thread',
            'is_wsb_daily_discussion',
        ]

    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        title = submission_data.get('title', '').lower()
        comments = submission_data.get('comments', [])
        all_text = f"{title} " + " ".join(comments).lower()
        
        # Count keywords by category
        pennystic = '🇹‌🇭‌🇪‌ 🇱‌🇴‌🇺‌🇳‌🇬‌🇪‌' in title
        earnings = 'upcoming earnings' in title
        theta_daily = 'daily r/thetagang discussion thread' in title
        daily_stock_discussion = 'daily general discussion and advice thread' in title
        daily_discussion_thread = 'daily discussion thread' in title
        wsb_daily = 'what are your moves tomorrow' in title
        
        return {
            'is_penny_stock_discussion': float(pennystic),
            'is_earnings_discussion': float(earnings),
            'is_theta_daily_discussion': float(theta_daily),
            'is_daily_stock_discussion': float(daily_stock_discussion),
            'is_daily_discussion_thread': float(daily_discussion_thread),
            'is_wsb_daily_discussion': float(wsb_daily),
        }


class KeywordFeaturizer(BaseFeaturizer):
    """Extract keyword-related features"""
    
    def __init__(self, keyword_file: Optional[Path] = None):
        self.stock_keywords = self._load_keywords(keyword_file)
    
    def get_feature_names(self) -> List[str]:
        return [
            'high_relevance_keywords',
            'medium_relevance_keywords', 
            'low_relevance_keywords',
            'total_keyword_count',
            'keyword_density',
            'title_keyword_count',
            'title_keyword_density',
            'unique_keyword_ratio'
        ]
    
    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        title = submission_data.get('title', '').lower()
        comments = submission_data.get('comments', [])
        all_text = f"{title} " + " ".join(comments).lower()
        
        # Count keywords by category
        high_count = sum(1 for kw in self.stock_keywords['high'] if kw in all_text)
        medium_count = sum(1 for kw in self.stock_keywords['medium'] if kw in all_text)
        low_count = sum(1 for kw in self.stock_keywords['low'] if kw in all_text)
        
        # Title keywords
        title_high = sum(1 for kw in self.stock_keywords['high'] if kw in title)
        title_medium = sum(1 for kw in self.stock_keywords['medium'] if kw in title)
        title_low = sum(1 for kw in self.stock_keywords['low'] if kw in title)
        title_keyword_count = title_high + title_medium + title_low
        
        # Unique keywords found
        all_keywords = self.stock_keywords['high'] | self.stock_keywords['medium'] | self.stock_keywords['low']
        unique_keywords_found = sum(1 for kw in all_keywords if kw in all_text)
        
        # Density calculations
        word_count = len(all_text.split())
        title_word_count = len(title.split())
        total_keywords = high_count + medium_count + low_count
        
        return {
            'high_relevance_keywords': float(high_count),
            'medium_relevance_keywords': float(medium_count),
            'low_relevance_keywords': float(low_count),
            'total_keyword_count': float(total_keywords),
            'keyword_density': (total_keywords / max(word_count, 1)) * 100,
            'title_keyword_count': float(title_keyword_count),
            'title_keyword_density': (title_keyword_count / max(title_word_count, 1)) * 100,
            'unique_keyword_ratio': unique_keywords_found / max(len(all_keywords), 1)
        }
    
    def _load_keywords(self, keyword_file: Optional[Path]) -> Dict[str, set]:
        """Load keywords from file or use defaults"""
        if keyword_file and keyword_file.exists():
            with open(keyword_file, 'r') as f:
                return json.load(f)
        
        return {
            'high': {
                'buy', 'sell', 'hold', 'calls', 'puts', 'options', 'shares', 'stock', 'stocks',
                'position', 'portfolio', 'investment', 'trading', 'ticker', 'earnings',
                'dividend', 'bull', 'bear', 'bullish', 'bearish', 'long', 'short'
            },
            'medium': {
                'company', 'business', 'financial', 'finance', 'money', 'price', 'value',
                'growth', 'analyst', 'forecast', 'target', 'upgrade', 'downgrade'
            },
            'low': {
                'market', 'economy', 'economic', 'news', 'report', 'quarter', 'annual'
            }
        }

class PsychologyFeaturizer(BaseFeaturizer):
    """Extract psychology/advice features (negative signals)"""
    
    def get_feature_names(self) -> List[str]:
        return [
            'psychology_keyword_count',
            'advice_pattern_count',
            'question_count',
            'help_seeking_count',
            'uncertainty_indicators',
            'emotional_language_count'
        ]
    
    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        title = submission_data.get('title', '').lower()
        comments = submission_data.get('comments', [])
        all_text = f"{title} " + " ".join(comments).lower()
        
        # Psychology keywords
        psychology_keywords = {
            'psychology', 'psychological', 'emotion', 'emotional', 'fear', 'greed',
            'anxiety', 'stress', 'worried', 'nervous', 'confidence', 'doubt'
        }
        psych_count = sum(1 for kw in psychology_keywords if kw in all_text)
        
        # Advice patterns
        advice_patterns = [
            r'should i', r'what should', r'how to', r'need help', r'advice',
            r'beginner', r'newbie', r'help me', r'can someone'
        ]
        advice_count = sum(1 for pattern in advice_patterns if re.search(pattern, all_text))
        
        # Questions
        question_count = all_text.count('?')
        
        # Help seeking
        help_patterns = [r'help', r'guidance', r'suggestions', r'recommendations']
        help_count = sum(1 for pattern in help_patterns if re.search(pattern, all_text))
        
        # Uncertainty indicators
        uncertainty_words = ['maybe', 'perhaps', 'might', 'could', 'unsure', 'confused']
        uncertainty_count = sum(1 for word in uncertainty_words if word in all_text)
        
        # Emotional language
        emotional_words = ['scared', 'excited', 'worried', 'happy', 'sad', 'angry', 'frustrated']
        emotional_count = sum(1 for word in emotional_words if word in all_text)
        
        return {
            'psychology_keyword_count': float(psych_count),
            'advice_pattern_count': float(advice_count),
            'question_count': float(question_count),
            'help_seeking_count': float(help_count),
            'uncertainty_indicators': float(uncertainty_count),
            'emotional_language_count': float(emotional_count)
        }

class StructureFeaturizer(BaseFeaturizer):
    """Extract content structure features"""
    
    def get_feature_names(self) -> List[str]:
        return [
            'title_length',
            'total_text_length',
            'comment_count',
            'avg_comment_length',
            'submission_score',
            'title_word_count',
            'text_to_title_ratio',
            'has_submission_body'
        ]
    
    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        title = submission_data.get('title', '')
        comments = submission_data.get('comments', [])
        metadata = submission_data.get('metadata', {})
        
        # Check if first comment is submission body
        has_body = False
        if comments and comments[0].startswith('Submission body:'):
            has_body = True
        
        # Calculate lengths
        total_text_length = len(title) + sum(len(c) for c in comments)
        avg_comment_length = sum(len(c) for c in comments) / max(len(comments), 1)
        
        return {
            'title_length': float(len(title)),
            'total_text_length': float(total_text_length),
            'comment_count': float(len(comments)),
            'avg_comment_length': avg_comment_length,
            'submission_score': float(metadata.get('original_score', 0)),
            'title_word_count': float(len(title.split())),
            'text_to_title_ratio': total_text_length / max(len(title), 1),
            'has_submission_body': float(has_body)
        }


class EngagementFeaturizer(BaseFeaturizer):
    """Extract engagement-related features"""
    
    def get_feature_names(self) -> List[str]:
        return [
            'avg_comment_score',
            'max_comment_score',
            'min_comment_score',
            'high_score_comments',
            'negative_score_comments',
            'score_variance',
            'score_range',
            'positive_engagement_ratio'
        ]
    
    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        comments = submission_data.get('comments', [])
        
        # Extract scores from comments
        scores = []
        for comment in comments:
            score_match = re.search(r'\(score (-?\d+)\)', comment)
            if score_match:
                scores.append(int(score_match.group(1)))
        
        if not scores:
            return {name: 0.0 for name in self.get_feature_names()}
        
        scores_array = np.array(scores)
        
        return {
            'avg_comment_score': float(np.mean(scores_array)),
            'max_comment_score': float(np.max(scores_array)),
            'min_comment_score': float(np.min(scores_array)),
            'high_score_comments': float(np.sum(scores_array >= 5)),
            'negative_score_comments': float(np.sum(scores_array < 0)),
            'score_variance': float(np.var(scores_array)),
            'score_range': float(np.max(scores_array) - np.min(scores_array)),
            'positive_engagement_ratio': float(np.sum(scores_array > 0) / len(scores_array))
        }

class TextQualityFeaturizer(BaseFeaturizer):
    """Extract text quality features"""
    
    def get_feature_names(self) -> List[str]:
        return [
            'uppercase_ratio',
            'punctuation_density',
            'avg_word_length',
            'unique_word_ratio',
            'sentence_count',
            'avg_sentence_length',
            'exclamation_count',
            'caps_word_count'
        ]
    
    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        title = submission_data.get('title', '')
        comments = submission_data.get('comments', [])
        all_text = f"{title} " + " ".join(comments)
        
        if not all_text.strip():
            return {name: 0.0 for name in self.get_feature_names()}
        
        # Basic text statistics
        uppercase_chars = sum(1 for c in all_text if c.isupper())
        punctuation_chars = sum(1 for c in all_text if c in '.,!?;:')
        
        words = all_text.split()
        sentences = re.split(r'[.!?]+', all_text)
        
        # CAPS words (all uppercase words with length > 1)
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        return {
            'uppercase_ratio': uppercase_chars / len(all_text),
            'punctuation_density': punctuation_chars / len(all_text),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
            'sentence_count': float(len([s for s in sentences if s.strip()])),
            'avg_sentence_length': len(all_text) / max(len([s for s in sentences if s.strip()]), 1),
            'exclamation_count': float(all_text.count('!')),
            'caps_word_count': float(caps_words)
        }
    

class SlidingWindowEmbeddingFeaturizer(BaseFeaturizer):
    def __init__(self,
                 model_name="ProsusAI/finbert",
                 window_tokens=256,
                 stride_tokens=64,
                 max_chars=1000,
                 device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(model_name).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mdl.to(self.device)

        self.W = window_tokens
        self.S = stride_tokens
        self.max_chars = max_chars
        self.dim = self.mdl.config.hidden_size
        self._names = [f"embed_{i}" for i in range(self.dim)]

    # ---------------------------------------------------------------------
    def get_feature_names(self) -> List[str]:
        return self._names

    @torch.inference_mode()
    def extract_features(self, submission_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert the entire submission (title + comments) into one 768-d vector by:
        1. Tokenising once (no specials)                        → token_ids
        2. Slicing into overlapping windows  ≤ 510 tokens long → windows
        3. Adding CLS/SEP, padding, attention mask             → input_ids, attn
        4. Forward pass, take CLS of each window               → [N, 768]
        5. Mean-pool across windows                            → [768]
        Returns { "embed_0": float, … "embed_767": float }.
        """
        # -------- 1. flatten + early-exit
        title = submission_data.get("title", "")
        comments = submission_data.get("comments", [])
        text = (title + " " + " ".join(comments))[: self.max_chars]

        if not text.strip():
            return {name: 0.0 for name in self._names}

        token_ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            return {name: 0.0 for name in self._names}

        # -------- 2. build windows
        cls_id, sep_id = self.tok.cls_token_id, self.tok.sep_token_id
        pad_id = self.tok.pad_token_id
        max_model_len = self.tok.model_max_length          # 512 for FinBERT
        win_len = min(self.W, max_model_len - 2)           # reserve CLS+SEP
        stride = min(self.S, win_len)

        windows = [
            [cls_id] + token_ids[i : i + win_len] + [sep_id]
            for i in range(0, len(token_ids), stride)
        ] or [[cls_id, sep_id]]                            # guarantee ≥1 window

        # -------- 3. pad batch + attention mask
        max_len = max(len(w) for w in windows)
        assert max_len <= max_model_len, f"window={max_len} > {max_model_len}"

        input_ids = torch.full(
            (len(windows), max_len), pad_id, dtype=torch.long, device=self.device
        )
        attn = torch.zeros_like(input_ids)

        for i, w in enumerate(windows):
            input_ids[i, : len(w)] = torch.tensor(w, dtype=torch.long, device=self.device)
            attn[i, : len(w)] = 1

        # -------- 4. model forward → CLS per window
        cls_vecs = self.mdl(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0]

        # -------- 5. mean-pool windows → 1 vector
        doc_vec = cls_vecs.mean(dim=0).cpu().numpy()       # [768]

        return {name: float(v) for name, v in zip(self._names, doc_vec)}

def create_default_featurizer() -> SubmissionFeaturizer:
    """Create featurizer with default extractors"""
    return SubmissionFeaturizer()

def create_finbert_featurizer() -> SubmissionFeaturizer:
    """Create featurizer with FinBERT extractors"""
    return SubmissionFeaturizer([
        SlidingWindowEmbeddingFeaturizer()
    ])