import re
import html
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class TextCleaningConfig:
    """Configuration for text cleaning operations"""
    remove_urls: bool = True
    remove_reddit_markup: bool = True
    remove_user_mentions: bool = False
    remove_subreddit_mentions: bool = False
    decode_html_entities: bool = True
    normalize_whitespace: bool = True
    max_length: Optional[int] = None
    remove_deleted_markers: bool = True

class CommentTextProcessor:
    """
    Handles all text processing operations for Reddit comments.
    
    Single Responsibility: Text manipulation and analysis only.
    No knowledge of Reddit API, file storage, or data formatting.
    """
    
    def __init__(self, config: TextCleaningConfig = None):
        self.config = config or TextCleaningConfig()
        
        # Pre-compile regex patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile frequently used regex patterns"""
        self.patterns = {
            'urls': re.compile(r'https?://\S+|www\.\S+'),
            'reddit_images': re.compile(r'!\[img\]\(emote\|[^)]+\)'),
            'reddit_links': re.compile(r'\[([^\]]+)\]\([^)]+\)'),
            'user_mentions': re.compile(r'/u/\w+|u/\w+'),
            'subreddit_mentions': re.compile(r'/r/\w+|r/\w+'),
            'multiple_whitespace': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n+'),
            'deleted_markers': re.compile(r'\[deleted\]|\[removed\]', re.IGNORECASE)
        }
    
    def clean_text(self, text: str) -> str:
        """
        Main text cleaning method
        
        Args:
            text: Raw comment text
            
        Returns:
            Cleaned text according to configuration
        """
        if not text or not isinstance(text, str):
            return ""
        
        cleaned = text
        
        # Apply cleaning operations based on configuration
        if self.config.decode_html_entities:
            cleaned = self._decode_html_entities(cleaned)
        
        if self.config.remove_urls:
            cleaned = self._remove_urls(cleaned)
        
        if self.config.remove_reddit_markup:
            cleaned = self._remove_reddit_markup(cleaned)
        
        if self.config.remove_user_mentions:
            cleaned = self._remove_user_mentions(cleaned)
        
        if self.config.remove_subreddit_mentions:
            cleaned = self._remove_subreddit_mentions(cleaned)
        
        if self.config.remove_deleted_markers:
            cleaned = self._remove_deleted_markers(cleaned)
        
        if self.config.normalize_whitespace:
            cleaned = self._normalize_whitespace(cleaned)
        
        if self.config.max_length:
            cleaned = self._truncate_text(cleaned, self.config.max_length)
        
        return cleaned.strip()
    
    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities like &amp; &lt; &gt;"""
        return html.unescape(text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs"""
        return self.patterns['urls'].sub('', text)
    
    def _remove_reddit_markup(self, text: str) -> str:
        """Remove Reddit-specific markup like image embeds and links"""
        # Remove image embeds
        text = self.patterns['reddit_images'].sub('', text)
        # Convert [text](url) to just text
        text = self.patterns['reddit_links'].sub(r'\1', text)
        return text
    
    def _remove_user_mentions(self, text: str) -> str:
        """Remove user mentions like /u/username"""
        return self.patterns['user_mentions'].sub('', text)
    
    def _remove_subreddit_mentions(self, text: str) -> str:
        """Remove subreddit mentions like /r/subreddit"""
        return self.patterns['subreddit_mentions'].sub('', text)
    
    def _remove_deleted_markers(self, text: str) -> str:
        """Remove [deleted] and [removed] markers"""
        return self.patterns['deleted_markers'].sub('', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and newlines"""
        # Replace multiple whitespace with single space
        text = self.patterns['multiple_whitespace'].sub(' ', text)
        # Replace multiple newlines with single newline
        text = self.patterns['multiple_newlines'].sub('\n', text)
        return text
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def calculate_comment_depth(self, comment: Dict[str, Any], all_comments: List[Dict[str, Any]]) -> int:
        """
        Calculate the depth of a comment in the thread hierarchy
        
        Args:
            comment: Comment dictionary with 'parent_id' field
            all_comments: List of all comments to traverse hierarchy
            
        Returns:
            Depth level (0 for top-level, 1 for reply to top-level, etc.)
        """
        if not comment.get('parent_id'):
            return 0
        
        # Create lookup map for performance
        comment_map = {c['id']: c for c in all_comments if 'id' in c}
        
        depth = 0
        current_parent_id = comment.get('parent_id', '')
        
        # Remove 't1_' or 't3_' prefix if present (Reddit thing IDs)
        if current_parent_id.startswith('t1_'):
            current_parent_id = current_parent_id[3:]
        elif current_parent_id.startswith('t3_'):
            # t3_ means it's a direct reply to submission, not another comment
            return 1
        
        # Traverse up the parent chain
        while current_parent_id and depth < 50:  # Prevent infinite loops
            parent_comment = comment_map.get(current_parent_id)
            if not parent_comment:
                break
            
            depth += 1
            current_parent_id = parent_comment.get('parent_id', '')
            
            if current_parent_id.startswith('t1_'):
                current_parent_id = current_parent_id[3:]
            elif current_parent_id.startswith('t3_'):
                depth += 1  # One more for the submission
                break
        
        return depth
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extract various text features for analysis
        
        Args:
            text: Comment text
            
        Returns:
            Dictionary with text features
        """
        if not text:
            return {}
        
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'contains_urls': bool(self.patterns['urls'].search(text)),
            'contains_user_mentions': bool(self.patterns['user_mentions'].search(text)),
            'contains_subreddit_mentions': bool(self.patterns['subreddit_mentions'].search(text)),
            'line_count': len(text.split('\n')),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'punctuation_count': sum(1 for c in text if c in '!?.,;:'),
        }
    
    def is_low_quality_comment(self, text: str, min_words: int = 3) -> bool:
        """
        Determine if a comment is low quality based on text analysis
        
        Args:
            text: Comment text
            min_words: Minimum word count threshold
            
        Returns:
            True if comment appears to be low quality
        """
        if not text or len(text.strip()) == 0:
            return True
        
        # Check for common low-quality patterns
        low_quality_patterns = [
            r'^(lol|lmao|wtf|omg|this|nice|cool|wow|ok|okay)\.?$',
            r'^\w{1,2}$',  # Single or two characters
            r'^[\W_]+$',   # Only punctuation/symbols
        ]
        
        for pattern in low_quality_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        
        # Check word count
        word_count = len(text.split())
        if word_count < min_words:
            return True
        
        return False