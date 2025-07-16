from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CommentAggregator:
    """
    Pure comment aggregation - builds comment-reply chains
    
    Single Responsibility: Transform flat comment list into hierarchical comment threads
    No text processing, no submission body handling - just pure structural aggregation
    """
    
    def __init__(self):
        """No dependencies - pure aggregation logic"""
        pass
    
    def aggregate_comments_to_threads(self, comments: List[Dict[str, Any]]) -> List[str]:
        """
        Main method: Convert flat comment list to threaded comment strings
        
        Args:
            comments: Raw comments from Reddit API (body text should be pre-processed if desired)
            
        Returns:
            List of formatted comment thread strings: ["Top level comment: A; Replies: B, C", ...]
        """
        if not comments:
            return []
        
        # Step 1: Build comment hierarchy 
        comment_hierarchy = self.build_comment_hierarchy(comments)
        
        # Step 2: Format each thread into strings
        formatted_threads = []
        
        for thread_data in comment_hierarchy.values():
            parent_comment = thread_data['parent']
            replies = thread_data['replies']
            
            formatted_thread = self.format_comment_thread(parent_comment, replies)
            if formatted_thread:  # Only add non-empty threads
                formatted_threads.append(formatted_thread)
        
        logger.info(f"Aggregated {len(comments)} raw comments into {len(formatted_threads)} comment threads")
        return formatted_threads
    
    def build_comment_hierarchy(self, comments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Build comment hierarchy by grouping replies under their parent comments
        
        Args:
            comments: Raw comments from Reddit API
            
        Returns:
            Dictionary mapping parent comment IDs to their thread data:
            {
                'parent_comment_id': {
                    'parent': parent_comment_dict,
                    'replies': [list_of_reply_dicts]
                }
            }
        """
        # Create lookup map for fast access
        comment_map = {comment['id']: comment for comment in comments}
        
        # Separate top-level comments from replies
        top_level_comments = []
        replies_map = {}  # parent_id -> list of replies
        
        for comment in comments:
            parent_id = comment.get('parent_id', '')
            
            if parent_id.startswith('t3_'):
                # This is a top-level comment (direct reply to submission)
                top_level_comments.append(comment)
            elif parent_id.startswith('t1_'):
                # This is a reply to another comment
                # Remove the 't1_' prefix to get the actual comment ID
                actual_parent_id = parent_id[3:]
                
                if actual_parent_id not in replies_map:
                    replies_map[actual_parent_id] = []
                replies_map[actual_parent_id].append(comment)
        
        # Build the final hierarchy using comment IDs as keys
        hierarchy = {}
        
        for top_comment in top_level_comments:
            comment_id = top_comment['id']
            # Get all replies for this top-level comment (including nested replies)
            all_replies = self._get_all_replies_recursive(comment_id, replies_map, comment_map)
            
            hierarchy[comment_id] = {
                'parent': top_comment,
                'replies': all_replies
            }
        
        logger.debug(f"Built hierarchy: {len(top_level_comments)} top-level comments, "
                    f"{sum(len(thread['replies']) for thread in hierarchy.values())} total replies")
        
        return hierarchy
    
    def _get_all_replies_recursive(self, comment_id: str, replies_map: Dict[str, List[Dict]], 
                                 comment_map: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Recursively get all replies to a comment (including replies to replies)
        
        Args:
            comment_id: The comment ID to find replies for
            replies_map: Map of parent_id -> replies list
            comment_map: Map of comment_id -> comment data
            
        Returns:
            Flattened list of all replies in the thread
        """
        all_replies = []
        direct_replies = replies_map.get(comment_id, [])
        
        for reply in direct_replies:
            all_replies.append(reply)
            # Recursively get replies to this reply
            nested_replies = self._get_all_replies_recursive(reply['id'], replies_map, comment_map)
            all_replies.extend(nested_replies)
        
        return all_replies
    
    def format_comment_thread(self, parent_comment: Dict[str, Any], 
                            replies: List[Dict[str, Any]]) -> str:
        """
        Format a comment thread into the required string format:
        'Top level comment: A; Replies: B, C'
        
        Args:
            parent_comment: The top-level comment
            replies: List of all replies to this comment
            
        Returns:
            Formatted thread string or empty string if comment has no body
        """
        # Get parent comment text (no cleaning - use as-is)
        parent_text = parent_comment.get('body', '').strip()
        
        if not parent_text:
            return ""  # Skip empty comments
        
        # Start with the top-level comment
        parent_score = parent_comment.get('score', 0)
        formatted_thread = f"Top level comment: {parent_text} (score {parent_score})"
        
        # Add replies if they exist
        if replies:
            reply_texts = []
            for reply in replies:
                reply_text = reply.get('body', '').strip()
                if reply_text:  # Only include non-empty replies
                    reply_score = reply.get('score', 0)
                    reply_with_score = f"{reply_text} (score {reply_score})"
                    reply_texts.append(reply_with_score)
            
            if reply_texts:
                replies_string = "; ".join(reply_texts)
                formatted_thread += f"; Replies: {replies_string}"
        
        return formatted_thread
    
    def get_comment_statistics(self, comments: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get statistics about the comment structure
        
        Args:
            comments: Raw comments from Reddit API
            
        Returns:
            Dictionary with comment statistics
        """
        if not comments:
            return {'total': 0, 'top_level': 0, 'replies': 0, 'empty': 0, 'threads': 0}
        
        hierarchy = self.build_comment_hierarchy(comments)
        
        total_comments = len(comments)
        top_level_count = len(hierarchy)
        replies_count = sum(len(thread['replies']) for thread in hierarchy.values())
        empty_count = sum(1 for comment in comments if not comment.get('body', '').strip())
        
        return {
            'total': total_comments,
            'top_level': top_level_count, 
            'replies': replies_count,
            'empty': empty_count,
            'threads': len(hierarchy)
        }
    
    def get_comment_hierarchy_debug(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Debug method to inspect comment hierarchy structure
        
        Args:
            comments: Raw comments from Reddit API
            
        Returns:
            Debug information about the hierarchy
        """
        hierarchy = self.build_comment_hierarchy(comments)
        
        debug_info = {
            'total_threads': len(hierarchy),
            'threads': []
        }
        
        for comment_id, thread_data in hierarchy.items():
            parent_comment = thread_data['parent']
            replies = thread_data['replies']
            
            thread_info = {
                'parent_id': comment_id,
                'parent_body_preview': parent_comment.get('body', '')[:50] + '...' if parent_comment.get('body', '') else '[empty]',
                'reply_count': len(replies),
                'reply_ids': [reply.get('id') for reply in replies]
            }
            debug_info['threads'].append(thread_info)
        
        return debug_info