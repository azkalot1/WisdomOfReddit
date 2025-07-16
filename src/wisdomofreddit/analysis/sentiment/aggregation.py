from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from statistics import mean, median
import logging
from .state import SentimentResults, TickerAggregateStats, DailyTickerConsensus




class TickerSentimentAggregator:
    """Aggregates submission-level deduplicated sentiments into ticker-level consensus"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)


    def aggregate_daily_sentiments(self, 
                                sub_deduplicated_sentiments: List[SentimentResults],
                                date: str) -> DailyTickerConsensus:
        """
        Main aggregation method
        
        Args:
            sub_deduplicated_sentiments: List of deduplicated SentimentResults from submissions
            date: Date string for this batch
            
        Returns:
            DailyTickerConsensus with ticker-level aggregated data
        """
        if not sub_deduplicated_sentiments:
            self.logger.warning("No extraction results provided for aggregation")
            return self._create_empty_consensus(date)
        
        # Filter out SentimentResults with empty sentiments
        valid_results = [
            result for result in sub_deduplicated_sentiments 
            if result.sentiments  # Skip if sentiments list is empty
        ]
        
        if not valid_results:
            self.logger.warning(f"No valid sentiment results found for {date} (all had empty sentiments)")
            return self._create_empty_consensus(date)
        
        self.logger.debug(f"Aggregating {len(valid_results)} valid results (filtered from {len(sub_deduplicated_sentiments)})")
        
        # Step 1: Group all sentiments by ticker
        ticker_sentiments = self._group_sentiments_by_ticker(valid_results)
        
        if not ticker_sentiments:
            self.logger.warning(f"No ticker sentiments found after grouping for {date}")
            return self._create_empty_consensus(date)
        
        # Step 2: Calculate aggregate stats for each ticker
        ticker_stats = {}
        for ticker, sentiment_data in ticker_sentiments.items():
            stats = self._calculate_ticker_stats(ticker, sentiment_data)
            # Only include tickers that have actual data
            if stats.total_mentions > 0:
                ticker_stats[ticker] = stats
            else:
                self.logger.debug(f"Skipping ticker {ticker} with no mentions")
        
        if not ticker_stats:
            self.logger.warning(f"No valid ticker stats generated for {date}")
            return self._create_empty_consensus(date)
        
        # Step 3: Create consolidated result
        consolidated_result = DailyTickerConsensus(
            date=date,
            ticker_stats=ticker_stats,
            total_submissions_processed=len(valid_results),  # Use valid_results count
            total_sentiments_processed=sum(len(result.sentiments) for result in valid_results),
            unique_tickers=len(ticker_stats)
        )
        
        self.logger.info(f"Aggregation complete for {date}: "
                f"{len(valid_results)} valid submissions → "
                f"{len(ticker_stats)} unique tickers, "
                f"{consolidated_result.total_sentiments_processed} total sentiments")
        
        return consolidated_result
    
    
    def _group_sentiments_by_ticker(self, 
                                   sub_deduplicated_sentiments: List[SentimentResults]) -> Dict[str, Dict]:
        """
        Extract individual sentiments and group by ticker
        
        Returns:
            Dict[ticker, {
                'sentiments': List[Sentiment],
                'primary_count': int,  # How many times this was primary ticker
                'submission_ids': Set[str]  # If you track submission IDs
            }]
        """
        ticker_data = defaultdict(lambda: {
            'sentiments': [],
            'primary_count': 0,
            'submission_count': 0
        })
        
        for submission_result in sub_deduplicated_sentiments:
            submission_tickers = set()
            
            # Process each sentiment in this submission
            for sentiment in submission_result.sentiments:
                if sentiment.ticker and not sentiment.extraction_refusal:
                    normalized_ticker = sentiment.ticker.upper().strip()
                    ticker_data[normalized_ticker]['sentiments'].append(sentiment)
                    submission_tickers.add(normalized_ticker)
            
            # Count primary ticker occurrences
            if submission_result.primary_ticker:
                normalized_primary = submission_result.primary_ticker.upper().strip()
                ticker_data[normalized_primary]['primary_count'] += 1
            
            # Count unique submissions per ticker
            for ticker in submission_tickers:
                ticker_data[ticker]['submission_count'] += 1
        
        return dict(ticker_data)
    
    def _create_empty_consensus(self, date: str) -> DailyTickerConsensus:
        """Create empty consensus when no valid data is found"""
        return DailyTickerConsensus(
            date=date,
            ticker_stats={},
            total_submissions_processed=0,
            total_sentiments_processed=0,
            unique_tickers=0
        )
    
    def _create_empty_ticker_stats(self, ticker: str) -> TickerAggregateStats:
        """Create empty ticker stats when no sentiments are found"""
        return TickerAggregateStats(
            ticker=ticker,
            total_mentions=0,
            sentiment_distribution={},
            avg_confidence=0.0,
            avg_sentiment_intensity=0.0,
            confidence_weighted_intensity=0.0,
            position_distribution={},
            conviction_distribution={},
            time_horizon_distribution={},
            source_distribution={},
            unique_submissions=0,
            submissions_as_primary=0,
            explicit_mentions=0,
            implicit_mentions=0,
            avg_confidence_by_sentiment={},
            sentiment_consensus_strength=0.0,
            dominant_sentiment='neutral',  # Default
            dominant_sentiment_percentage=0.0
        )
    
    def _calculate_ticker_stats(self, ticker: str, sentiment_data: Dict) -> TickerAggregateStats:
        """Calculate comprehensive stats for a single ticker"""
        
        sentiments = sentiment_data['sentiments']
        
        if not sentiments:
            return self._create_empty_ticker_stats(ticker)
        
        # Basic counts
        total_mentions = len(sentiments)
        sentiment_counts = Counter(s.sentiment for s in sentiments)
        
        # Weighted averages
        total_confidence_weight = sum(s.confidence for s in sentiments)
        avg_confidence = mean(s.confidence for s in sentiments)
        avg_intensity = mean(s.sentiment_intensity for s in sentiments)
        confidence_weighted_intensity = sum(s.sentiment_intensity * s.confidence for s in sentiments) / total_confidence_weight
        
        # Categorical distributions
        position_dist = dict(Counter(s.position for s in sentiments))
        conviction_dist = dict(Counter(s.conviction for s in sentiments))
        time_horizon_dist = dict(Counter(s.time_horizon for s in sentiments))
        source_dist = dict(Counter(s.source for s in sentiments))
        
        # Quality metrics
        explicit_count = sum(1 for s in sentiments if not s.is_implicit)
        implicit_count = sum(1 for s in sentiments if s.is_implicit)
        
        # Confidence by sentiment type
        confidence_by_sentiment = {}
        for sentiment_type in sentiment_counts.keys():
            sentiment_confidences = [s.confidence for s in sentiments if s.sentiment == sentiment_type]
            confidence_by_sentiment[sentiment_type] = mean(sentiment_confidences)
        
        # 🎯 NEW: Conviction-weighted bullish/bearish scores
        conviction_weighted_bullish_score = 0.0
        conviction_weighted_bearish_score = 0.0
        
        # Separate bullish and bearish sentiments for average conviction calculation
        bullish_sentiments = [s for s in sentiments if s.sentiment == 'bullish']
        bearish_sentiments = [s for s in sentiments if s.sentiment == 'bearish']
        
        # Calculate conviction weights (convert conviction levels to numeric values)
        conviction_weights = {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0
        }
        
        # Calculate weighted scores for bullish sentiments
        for sentiment in bullish_sentiments:
            conviction_weight = conviction_weights.get(sentiment.conviction, 1.0)
            # Triple weighting: confidence × intensity × conviction
            weighted_score = sentiment.confidence * sentiment.sentiment_intensity * conviction_weight
            conviction_weighted_bullish_score += weighted_score
        
        # Calculate weighted scores for bearish sentiments
        for sentiment in bearish_sentiments:
            conviction_weight = conviction_weights.get(sentiment.conviction, 1.0)
            # Triple weighting: confidence × intensity × conviction
            weighted_score = sentiment.confidence * sentiment.sentiment_intensity * conviction_weight
            conviction_weighted_bearish_score += weighted_score
        
        # Calculate average conviction levels for bullish/bearish
        avg_bullish_conviction = 0.0
        avg_bearish_conviction = 0.0
        
        if bullish_sentiments:
            bullish_conviction_values = [conviction_weights.get(s.conviction, 1.0) for s in bullish_sentiments]
            avg_bullish_conviction = mean(bullish_conviction_values)
        
        if bearish_sentiments:
            bearish_conviction_values = [conviction_weights.get(s.conviction, 1.0) for s in bearish_sentiments]
            avg_bearish_conviction = mean(bearish_conviction_values)
        
        # 🎯 NEW: Polarization and controversy metrics
        sentiment_polarization = self._calculate_sentiment_polarization(sentiment_counts, total_mentions)
        sentiment_controversy_score = self._calculate_controversy_score(
            bullish_sentiments, bearish_sentiments, total_mentions
        )
        is_controversial = sentiment_controversy_score > 0.6  # Threshold for controversy
        
        # Consensus metrics
        dominant_sentiment = sentiment_counts.most_common(1)[0][0]
        dominant_count = sentiment_counts[dominant_sentiment]
        dominant_percentage = (dominant_count / total_mentions) * 100
        
        # Consensus strength: how unified the sentiment is (0-1)
        sentiment_consensus_strength = dominant_count / total_mentions
        
        return TickerAggregateStats(
            ticker=ticker,
            total_mentions=total_mentions,
            sentiment_distribution=dict(sentiment_counts),
            avg_confidence=round(avg_confidence, 2),
            avg_sentiment_intensity=round(avg_intensity, 2),
            confidence_weighted_intensity=round(confidence_weighted_intensity, 2),
            position_distribution=position_dist,
            conviction_distribution=conviction_dist,
            time_horizon_distribution=time_horizon_dist,
            source_distribution=source_dist,
            unique_submissions=sentiment_data['submission_count'],
            submissions_as_primary=sentiment_data['primary_count'],
            explicit_mentions=explicit_count,
            implicit_mentions=implicit_count,
            avg_confidence_by_sentiment=confidence_by_sentiment,
            sentiment_consensus_strength=round(sentiment_consensus_strength, 2),
            dominant_sentiment=dominant_sentiment,
            dominant_sentiment_percentage=round(dominant_percentage, 1),
            # 🎯 NEW FIELDS
            sentiment_polarization=round(sentiment_polarization, 3),
            sentiment_controversy_score=round(sentiment_controversy_score, 3),
            is_controversial=is_controversial,
            conviction_weighted_bullish_score=round(conviction_weighted_bullish_score, 2),
            conviction_weighted_bearish_score=round(conviction_weighted_bearish_score, 2),
            avg_bullish_conviction=round(avg_bullish_conviction, 2),
            avg_bearish_conviction=round(avg_bearish_conviction, 2)
        )

    def _calculate_sentiment_polarization(self, sentiment_counts: Counter, total_mentions: int) -> float:
        """
        Calculate sentiment polarization (0-1).
        Higher values indicate more polarized sentiment (strong bullish vs bearish).
        Lower values indicate more consensus or neutral sentiment.
        """
        if total_mentions == 0:
            return 0.0
        
        bullish_ratio = sentiment_counts.get('bullish', 0) / total_mentions
        bearish_ratio = sentiment_counts.get('bearish', 0) / total_mentions
        neutral_ratio = sentiment_counts.get('neutral', 0) / total_mentions
        
        # Polarization is high when bullish and bearish are both significant
        # and neutral is low
        polarization = (bullish_ratio * bearish_ratio) * 4  # Scale to 0-1
        
        # Reduce polarization if there's a lot of neutral sentiment
        polarization *= (1 - neutral_ratio * 0.5)
        
        return min(polarization, 1.0)

    def _calculate_controversy_score(self, bullish_sentiments: List, bearish_sentiments: List, total_mentions: int) -> float:
        """
        Calculate controversy score based on the strength and conviction of opposing sentiments.
        High controversy = strong bullish AND strong bearish sentiments with high conviction.
        """
        if total_mentions == 0:
            return 0.0
        
        bullish_count = len(bullish_sentiments)
        bearish_count = len(bearish_sentiments)
        
        if bullish_count == 0 or bearish_count == 0:
            return 0.0  # No controversy if only one side
        
        # Calculate average conviction strength for each side
        conviction_weights = {'low': 1.0, 'medium': 2.0, 'high': 3.0}
        
        bullish_conviction_avg = mean([
            conviction_weights.get(s.conviction, 1.0) for s in bullish_sentiments
        ])
        bearish_conviction_avg = mean([
            conviction_weights.get(s.conviction, 1.0) for s in bearish_sentiments
        ])
        
        # Controversy is higher when:
        # 1. Both sides have significant representation
        # 2. Both sides have high conviction
        # 3. The split is relatively even
        
        # Balance factor: how evenly split the sentiment is
        total_polar = bullish_count + bearish_count
        balance_factor = min(bullish_count, bearish_count) / max(bullish_count, bearish_count)
        
        # Conviction factor: average conviction of both sides
        conviction_factor = (bullish_conviction_avg + bearish_conviction_avg) / 6.0  # Normalize to 0-1
        
        # Representation factor: how much of total mentions are polar (not neutral)
        representation_factor = total_polar / total_mentions
        
        controversy_score = balance_factor * conviction_factor * representation_factor
        
        return min(controversy_score, 1.0)