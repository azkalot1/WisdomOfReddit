from internetwisdom.reddit.comment_processing import RedditCSVConverter, CSVConverterConfig
from pathlib import Path
import logging

def main():
    """Main conversion script"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = CSVConverterConfig(
        input_directory=Path("..\\data\\reddit_dump\\"),  # Where your CSV files are
        output_directory=Path("..\\reddit_comments"),  # Where to save JSON files
        subreddits=[
            "ai_trading",
            "investing", 
            "MemeStockMarket",
            "options",
            "OptionsMillionaire",
            "pennystocks",
            "RobinHoodPennyStocks",
            "StockInvest",
            "StockMarket",
            "stocks",
            "StocksAndTrading",
            "StocksInFocus",
            "stockstobuytoday",
            "TheRaceTo10Million",
            "Trading",
            "ValueInvesting",
            "wallstreetbets",
            "wallstreetbets2"
        ],
        max_comments_per_submission=100,  # Limit comments per submission
        include_submission_body_in_comments=True,
        remove_orphan_comments=True,  # Remove comments without corresponding submissions
        organize_by_subreddit=False,  # Create subreddit subfolders
        folder_date_format='%Y%m%d',  # YYYYMMDD format for folders
        include_scores_in_text=True,  # Include scores in comment text
        min_year=2017,  # Only include data from 2020 onwards
        max_year=2024   # No upper limit (or set to 2024, etc.)
    )
    
    # Run conversion
    converter = RedditCSVConverter(config)
    stats = converter.convert_all_subreddits()
    
    # Print folder structure sample
    converter.print_folder_structure_sample()
    
    # Print final statistics
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Subreddits processed: {stats['subreddits_processed']}")
    print(f"Submissions processed: {stats['submissions_processed']}")
    print(f"Comments processed: {stats['comments_processed']}")
    print(f"JSON files created: {stats['files_created']}")
    print(f"Date folders created: {stats['date_folders_created']}")
    print(f"Errors: {stats['errors']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print("\nYear Filtering Results:")
    print(f"Submissions filtered by year: {stats['submissions_filtered_by_year']}")
    print(f"Comments filtered by year: {stats['comments_filtered_by_year']}")
    print(f"Submission year filtering rate: {stats['data_quality']['year_filtering_rate']['submissions']:.1f}%")
    print(f"Comment year filtering rate: {stats['data_quality']['year_filtering_rate']['comments']:.1f}%")
    print("\nData Cleaning Results:")
    print(f"Submissions removed ([removed]/[deleted]): {stats['submissions_removed']}")
    print(f"Comments removed ([removed]/[deleted]): {stats['comments_removed']}")
    print(f"Orphan comments removed: {stats['orphan_comments_removed']}")
    print(f"Date parse errors: {stats['date_parse_errors']}")
    print(f"Submission removal rate: {stats['data_quality']['submission_removal_rate']:.1f}%")
    print(f"Comment removal rate: {stats['data_quality']['comment_removal_rate']:.1f}%")
    print(f"Orphan removal rate: {stats['data_quality']['orphan_removal_rate']:.1f}%")
    print(f"Date parse error rate: {stats['data_quality']['date_parse_error_rate']:.1f}%")

if __name__ == "__main__":
    main()