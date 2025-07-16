#!/usr/bin/env python3
"""
Reddit Comment Parser - Entry Point Script

Usage:
    python parse.py --subreddits wallstreetbets StockMarket --time=24h --limit=100 --force-update
    python parse.py --subreddits all --time=240h --limit=50
    python parse.py --help
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List
from datetime import datetime
import dotenv
dotenv.load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import our Reddit processing modules
from internetwisdom.reddit import (
    PrawRedditClient,
    RedditConfigFactory,
    CommentTextProcessor,
    CommentAggregator,
    RedditCommentFormatter
)
from internetwisdom.data import JsonFileStorage, FilePathManager, StorageConfig
from internetwisdom import SubmissionProcessor, ProcessingResult

def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def setup_pipeline(output_dir: Path, comment_limit: int = 1000):
    """
    Initialize all components of the Reddit comment processing pipeline
    
    Args:
        output_dir: Directory to save reddit comments (e.g., ./reddit_comments)
        comment_limit: Maximum comments per submission
        
    Returns:
        tuple: (reddit_client, submission_processor)
    """
    
    # Configuration
    reddit_config = RedditConfigFactory.from_environment()
    storage_config = StorageConfig(
        base_directory=output_dir,
        organize_by_subreddit=False,    # YYYYMMDD/{submission_id}.json structure
        organize_by_date=True,
        date_format='%Y%m%d'
    )
    
    # Initialize components
    reddit_client = PrawRedditClient(reddit_config)
    aggregator = CommentAggregator()
    formatter = RedditCommentFormatter(aggregator)
    storage = JsonFileStorage(indent=2)
    path_manager = FilePathManager(storage_config)
    
    # Create submission processor
    processor = SubmissionProcessor(
        reddit_client=reddit_client,
        formatter=formatter,
        storage=storage,
        path_manager=path_manager,
        comment_limit=comment_limit
    )
    
    return reddit_client, processor

def parse_time_argument(time_str: str) -> int:
    """
    Parse time argument like '24h', '240h', '7d' into hours
    
    Args:
        time_str: Time string like '24h', '240h', '7d'
        
    Returns:
        Hours as integer
    """
    time_str = time_str.lower().strip()
    
    if time_str.endswith('h'):
        return int(time_str[:-1])
    elif time_str.endswith('d'):
        return int(time_str[:-1]) * 24
    else:
        # Assume hours if no unit
        return int(time_str)

def parse_subreddit_comments(reddit_client, 
                           processor: SubmissionProcessor,
                           subreddit: str, 
                           hours: int, 
                           limit: int,
                           force_update: bool,
                           inverse_order: bool,
                           ) -> List[ProcessingResult]:
    """
    Parse comments from a subreddit for the last N hours
    
    Args:
        reddit_client: Reddit client instance
        processor: Submission processor instance
        subreddit: Subreddit name
        hours: Time period in hours
        limit: Maximum submissions to fetch
        force_update: Overwrite existing files
        inverse_order: Process submissions in reverse order (oldest first)
        
    Returns:
        List of ProcessingResult objects
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting comment parsing for r/{subreddit} (last {hours} hours, limit {limit})")
    
    try:
        # Get submissions from timeframe
        logger.info(f"Fetching submission IDs for r/{subreddit}")
        submission_ids = reddit_client.get_submissions_by_timeframe(
            subreddit=subreddit,
            hours=hours,
            limit=limit,
            inverse_order=inverse_order
        )
        
        if not submission_ids:
            logger.warning(f"No submissions found for r/{subreddit} in last {hours} hours")
            return []
        
        logger.info(f"Found {len(submission_ids)} submissions to process")
        
        # Process all submissions
        results = processor.process_multiple_submissions(
            submission_ids=submission_ids,
            force_update=force_update,
            include_metadata=True
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to parse comments for r/{subreddit}: {e}")
        return []

def parse_multiple_subreddits(reddit_client,
                            processor: SubmissionProcessor,
                            subreddits: List[str], 
                            hours: int, 
                            limit: int,
                            force_update: bool,
                            inverse_order: bool
                            ) -> dict:
    """
    Parse comments from multiple subreddits
    
    Returns:
        dict: subreddit -> processing summary
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch parsing for {len(subreddits)} subreddits")
    
    all_results = {}
    
    for i, subreddit in enumerate(subreddits, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing subreddit {i}/{len(subreddits)}: r/{subreddit}")
        logger.info(f"{'='*60}")
        
        try:
            results = parse_subreddit_comments(
                reddit_client, processor, subreddit, hours, limit, force_update, inverse_order
            )
            
            # Calculate summary
            successful = len([r for r in results if r.success and not r.skipped])
            skipped = len([r for r in results if r.skipped])
            failed = len([r for r in results if not r.success])
            
            all_results[subreddit] = {
                'status': 'SUCCESS',
                'total': len(results),
                'processed': successful,
                'skipped': skipped,
                'failed': failed
            }
            
            print_subreddit_summary(subreddit, results)
            
        except Exception as e:
            logger.error(f"Failed to process r/{subreddit}: {e}")
            all_results[subreddit] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    return all_results

def print_subreddit_summary(subreddit: str, results: List[ProcessingResult]):
    """Print summary for a single subreddit"""
    
    successful = [r for r in results if r.success and not r.skipped]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if not r.success]
    
    print(f"\n📊 SUMMARY FOR r/{subreddit}")
    print(f"   Total submissions: {len(results)}")
    print(f"   ✅ Processed: {len(successful)}")
    print(f"   ⏩ Skipped: {len(skipped)}")
    print(f"   ❌ Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r.processing_time_seconds or 0 for r in successful) / len(successful)
        total_comments = sum(r.statistics.get('raw_comment_count', 0) for r in successful if r.statistics)
        print(f"   ⏱️  Avg time: {avg_time:.2f}s")
        print(f"   💬 Total comments: {total_comments}")

def print_final_summary(all_results: dict):
    """Print final summary across all subreddits"""
    
    print(f"\n{'='*60}")
    print("🎯 FINAL SUMMARY")
    print(f"{'='*60}")
    
    total_processed = 0
    total_failed = 0
    
    for subreddit, result in all_results.items():
        if result['status'] == 'SUCCESS':
            processed = result['processed']
            failed = result['failed']
            skipped = result['skipped']
            total_processed += processed
            total_failed += failed
            
            print(f"r/{subreddit:<20} ✅ {processed:>3} processed, ⏩ {skipped:>3} skipped, ❌ {failed:>3} failed")
        else:
            total_failed += 1
            print(f"r/{subreddit:<20} ❌ FAILED: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎯 TOTALS: {total_processed} submissions processed, {total_failed} failed")

def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Reddit Comment Parser - Fetch and save Reddit comments to JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse specific subreddits for last 24 hours
  python parse.py --subreddits wallstreetbets StockMarket --time=24h --limit=100
  
  # Parse all predefined subreddits for last 10 days with force update
  python parse.py --subreddits all --time=240h --force-update --limit=50
  
  # Parse single subreddit with verbose logging
  python parse.py --subreddits thetagang --time=48h --limit=200 --verbose
        """
    )
    
    parser.add_argument(
        '--subreddits', 
        nargs='+', 
        required=True,
        help='List of subreddits to parse, or "all" for predefined list'
    )
    
    parser.add_argument(
        '--time', 
        default='24h',
        help='Time period to look back (e.g., 24h, 240h, 7d). Default: 24h'
    )
    
    parser.add_argument(
        '--limit', 
        type=int, 
        default=100,
        help='Maximum number of submissions per subreddit. Default: 100'
    )
    
    parser.add_argument(
        '--force-update', 
        action='store_true',
        help='Overwrite existing files (default: skip existing files)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=Path,
        default=Path('./reddit_comments'),
        help='Output directory for JSON files. Default: ./reddit_comments'
    )
    parser.add_argument(
        '--inverse_order',
        action='store_true',
        help='Process submissions in reverse order (oldest first). Default: False'
    )
    
    parser.add_argument(
        '--comment-limit',
        type=int,
        default=1000,
        help='Maximum comments per submission. Default: 1000'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Parse time argument
    try:
        hours = parse_time_argument(args.time)
    except ValueError:
        logger.error(f"Invalid time format: {args.time}. Use formats like '24h', '240h', '7d'")
        sys.exit(1)
    
    # Parse subreddits
    if args.subreddits == ['all']:
        subreddits = get_predefined_subreddits()
    else:
        subreddits = args.subreddits
    
    # Print configuration
    logger.info(f"🚀 Starting Reddit Comment Parser")
    logger.info(f"   Subreddits: {', '.join(subreddits)}")
    logger.info(f"   Time period: {hours} hours")
    logger.info(f"   Submission limit: {args.limit} per subreddit")
    logger.info(f"   Comment limit: {args.comment_limit} per submission")
    logger.info(f"   Force update: {args.force_update}")
    logger.info(f"   Output directory: {args.output_dir}")
    logger.info(f"   Inverse order: {args.inverse_order}")
    
    # Setup pipeline
    try:
        reddit_client, processor = setup_pipeline(args.output_dir, args.comment_limit)
        
        # Test connection
        reddit_client.test_connection()
        logger.info("✅ Reddit connection successful")
        
    except Exception as e:
        logger.error(f"❌ Failed to setup pipeline: {e}")
        sys.exit(1)
    
    # Process subreddits
    try:
        if len(subreddits) == 1:
            # Single subreddit
            results = parse_subreddit_comments(
                reddit_client, processor, subreddits[0], hours, args.limit, args.force_update, args.inverse_order
            )
            print_subreddit_summary(subreddits[0], results)
        else:
            # Multiple subreddits
            all_results = parse_multiple_subreddits(
                reddit_client, processor, subreddits, hours, args.limit, args.force_update, args.inverse_order
            )
            print_final_summary(all_results)
        
        logger.info("🎉 Processing complete!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()