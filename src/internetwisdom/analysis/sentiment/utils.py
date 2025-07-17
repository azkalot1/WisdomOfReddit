from datetime import datetime
from typing import Dict, Any, Optional
from .state import RedditContent


def create_reddit_content(data: Dict[str, Any], max_n_comments: Optional[int] = None) -> RedditContent:
    """
    Create RedditContent from raw data dictionary.
    
    Handles the structure where comments[0] is submission body.
    """
    # Extract comments properly
    comments = []
    if 'comments' in data and len(data['comments']) > 1:
        # Skip first element (submission body) and take actual comments
        comments = data['comments'][1:]
        if max_n_comments is not None:
            comments = comments[:max_n_comments]
    
    return RedditContent(
        title=data.get('title', ''),
        submission_body=data.get('selftext') or data.get('comments', [''])[0],
        comments=comments,
        submission_id=data.get('submission_id'),
        created_utc=datetime.fromtimestamp(data['created_utc']) if 'created_utc' in data else None,
        subreddit=data.get('subreddit')
    )