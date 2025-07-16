from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
from statistics import mean, median
import math
from .state import SentimentResults, Sentiment, DeduplicationConfig, ConflictInfo, DeduplicationStrategy

logger = logging.getLogger(__name__)



class SentimentDeduplicator:
    """
    Deduplicates and aggregates sentiment results from multiple extractions within a submission
    
    Takes multiple SentimentResults (from submission + each comment) and produces
    a single consolidated SentimentResults representing the overall submission sentiment.
    """
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        """
        Initialize sentiment deduplicator
        
        Args:
            config: Configuration for deduplication behavior
        """
        self.config = config or DeduplicationConfig()
        self.conflicts_resolved: List[ConflictInfo] = []
        self.stats = {
            'total_extractions_processed': 0,
            'unique_tickers_found': 0,
            'conflicts_resolved': 0,
            'sentiments_filtered': 0,
            'extraction_refusals_excluded': 0,
            'implicit_tickers_found': 0,
            'primary_ticker_changes': 0
        }
        
        # Conviction level weights for weighted strategies
        self.conviction_weights = {
            'yolo': 10,
            'high': 8,
            'moderate': 6,
            'low': 4,
            'hedge': 3,
            'unclear': 2
        }
    
    def deduplicate(self, extraction_results: List['SentimentResults']) -> 'SentimentResults':
        """
        Main deduplication method
        
        Args:
            extraction_results: List of SentimentResults from multiple extractions
            
        Returns:
            Single consolidated SentimentResults
        """
        if not extraction_results:
            logger.warning("No extraction results provided for deduplication")
            return self._create_empty_result()
        
        logger.debug(f"Deduplicating {len(extraction_results)} extraction results")
        
        # Reset conflicts for this deduplication
        self.conflicts_resolved = []
        
        # Update stats
        self.stats['total_extractions_processed'] += len(extraction_results)
        
        # Step 1: Group all sentiments by ticker
        ticker_sentiments = self._group_sentiments_by_ticker(extraction_results)
        self.stats['unique_tickers_found'] = len(ticker_sentiments)
        
        # Step 2: Resolve conflicts for each ticker
        final_sentiments = []
        for ticker, sentiments in ticker_sentiments.items():
            if len(sentiments) == 1:
                # No conflict, use as-is (but still apply quality filters)
                filtered_sentiment = self._apply_quality_filters(sentiments[0])
                if filtered_sentiment:
                    final_sentiments.append(filtered_sentiment)
            else:
                # Conflict resolution needed
                resolved_sentiment = self._resolve_ticker_conflicts(ticker, sentiments)
                if resolved_sentiment:
                    final_sentiments.append(resolved_sentiment)
                    self.stats['conflicts_resolved'] += 1
        # Step 3: Determine primary ticker
        primary_ticker = self._determine_primary_ticker(extraction_results, final_sentiments)
        # Step 4: Create consolidated result
        consolidated_result = SentimentResults(
            sentiments=final_sentiments,
            primary_ticker=primary_ticker,
            metadata=self._create_metadata(extraction_results, final_sentiments)
        )
        
        logger.info(f"Deduplication complete: {len(extraction_results)} extractions → "
                   f"{len(final_sentiments)} unique sentiments, primary: {primary_ticker}")
        
        return consolidated_result
    
    def _group_sentiments_by_ticker(self, extraction_results: List['SentimentResults']) -> Dict[str, List['Sentiment']]:
        """
        Group all sentiments by ticker symbol
        
        Returns:
            Dictionary mapping ticker -> list of sentiments for that ticker
        """
        ticker_groups = defaultdict(list)
        
        for result in extraction_results:
            for sentiment in result.sentiments:
                # Skip extraction refusals if configured
                if self.config.exclude_extraction_refusals and sentiment.extraction_refusal:
                    self.stats['extraction_refusals_excluded'] += 1
                    continue
                
                if sentiment.ticker:  # Skip sentiments without ticker
                    # Normalize ticker (uppercase, remove $)
                    normalized_ticker = sentiment.ticker.upper().replace('$', '').strip()
                    if normalized_ticker:
                        # Track implicit tickers
                        if sentiment.is_implicit:
                            self.stats['implicit_tickers_found'] += 1
                        
                        # Create a copy with normalized ticker
                        normalized_sentiment = Sentiment(
                            confidence=sentiment.confidence,
                            ticker=normalized_ticker,
                            is_implicit=sentiment.is_implicit,
                            sentiment=sentiment.sentiment,
                            sentiment_intensity=sentiment.sentiment_intensity,
                            position=sentiment.position,
                            conviction=sentiment.conviction,
                            time_horizon=sentiment.time_horizon,
                            source=sentiment.source,
                            explanation=sentiment.explanation,
                            extraction_refusal=sentiment.extraction_refusal
                        )
                        ticker_groups[normalized_ticker].append(normalized_sentiment)
        
        logger.debug(f"Grouped sentiments: {dict((k, len(v)) for k, v in ticker_groups.items())}")
        return dict(ticker_groups)
    
    def _resolve_ticker_conflicts(self, ticker: str, sentiments: List['Sentiment']) -> Optional['Sentiment']:
        """
        Resolve conflicts for a single ticker
        
        Args:
            ticker: Ticker symbol
            sentiments: List of potentially conflicting sentiments for this ticker
            
        Returns:
            Single resolved sentiment for the ticker or None if filtered out
        """
        logger.debug(f"Resolving conflicts for {ticker}: {len(sentiments)} sentiments")
        
        # Apply quality filters first
        quality_sentiments = [s for s in sentiments if self._passes_quality_filter(s)]
        
        if not quality_sentiments:
            logger.debug(f"All sentiments for {ticker} filtered out due to quality thresholds")
            self.stats['sentiments_filtered'] += len(sentiments)
            return None
        
        # Prefer explicit tickers over implicit ones if configured
        if self.config.prefer_explicit_tickers:
            explicit_sentiments = [s for s in quality_sentiments if not s.is_implicit]
            if explicit_sentiments:
                quality_sentiments = explicit_sentiments
                logger.debug(f"Using {len(explicit_sentiments)} explicit sentiments for {ticker}")
        
        if len(quality_sentiments) == 1:
            return quality_sentiments[0]
        
        # Check if there's actually a conflict (different sentiment values)
        unique_sentiments = set(s.sentiment for s in quality_sentiments)
        
        if len(unique_sentiments) == 1:
            # Same sentiment, just different intensities/confidences - merge them
            resolved = self._merge_same_sentiments(quality_sentiments)
        else:
            # True conflict - apply resolution strategy
            resolved = self._apply_resolution_strategy(ticker, quality_sentiments)
        
        # Record conflict info
        confidence_spread = max(s.confidence for s in quality_sentiments) - min(s.confidence for s in quality_sentiments)
        conviction_levels = list(set(s.conviction for s in quality_sentiments))
        
        conflict_info = ConflictInfo(
            ticker=ticker,
            conflicting_sentiments=quality_sentiments,
            resolution_method=self.config.strategy.value,
            final_sentiment=resolved,
            confidence_spread=confidence_spread,
            conviction_levels=conviction_levels
        )
        self.conflicts_resolved.append(conflict_info)
        
        return resolved
    
    def _apply_resolution_strategy(self, ticker: str, sentiments: List['Sentiment']) -> 'Sentiment':
        """Apply the configured resolution strategy"""
        
        if self.config.strategy == DeduplicationStrategy.HIGHEST_CONFIDENCE:
            return self._apply_highest_confidence_resolution(sentiments)
        elif self.config.strategy == DeduplicationStrategy.CONFIDENCE_WEIGHTED:
            return self._apply_confidence_weighted_resolution(sentiments)
        elif self.config.strategy == DeduplicationStrategy.MAJORITY_VOTE:
            return self._apply_majority_vote_resolution(sentiments)
        elif self.config.strategy == DeduplicationStrategy.INTENSITY_WEIGHTED:
            return self._apply_intensity_weighted_resolution(sentiments)
        elif self.config.strategy == DeduplicationStrategy.CONVICTION_WEIGHTED:
            return self._apply_conviction_weighted_resolution(sentiments)
        else:
            logger.warning(f"Unknown strategy {self.config.strategy}, falling back to highest confidence")
            return self._apply_highest_confidence_resolution(sentiments)
    
    def _apply_confidence_weighted_resolution(self, sentiments: List['Sentiment']) -> 'Sentiment':
        """Apply confidence-weighted averaging to resolve conflicts"""
        
        # Group by sentiment type
        sentiment_groups = defaultdict(list)
        for s in sentiments:
            sentiment_groups[s.sentiment].append(s)
        
        # Calculate weighted scores for each sentiment type
        sentiment_scores = {}
        for sentiment_type, group in sentiment_groups.items():
            total_weight = sum(s.confidence for s in group)
            weighted_intensity = sum(s.sentiment_intensity * s.confidence for s in group) / total_weight
            avg_confidence = sum(s.confidence for s in group) / len(group)
            
            sentiment_scores[sentiment_type] = {
                'score': total_weight,  # Total confidence weight
                'intensity': weighted_intensity,
                'confidence': avg_confidence,
                'count': len(group),
                'sentiments': group
            }
        
        # Pick the sentiment type with highest total confidence weight
        winning_sentiment = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k]['score'])
        winner_data = sentiment_scores[winning_sentiment]
        
        # Create resolved sentiment by merging the winning group
        resolved = self._merge_sentiment_group(winner_data['sentiments'], "confidence_weighted")
        
        logger.debug(f"Confidence-weighted resolution for {sentiments[0].ticker}: "
                    f"{winning_sentiment} (score={winner_data['score']:.1f})")
        
        return resolved
    
    def _apply_conviction_weighted_resolution(self, sentiments: List['Sentiment']) -> 'Sentiment':
        """Apply conviction-weighted averaging to resolve conflicts"""
        
        # Group by sentiment type
        sentiment_groups = defaultdict(list)
        for s in sentiments:
            sentiment_groups[s.sentiment].append(s)
        
        # Calculate weighted scores for each sentiment type using conviction weights
        sentiment_scores = {}
        for sentiment_type, group in sentiment_groups.items():
            total_weight = sum(self.conviction_weights.get(s.conviction, 1) for s in group)
            weighted_intensity = sum(s.sentiment_intensity * self.conviction_weights.get(s.conviction, 1) for s in group) / total_weight
            weighted_confidence = sum(s.confidence * self.conviction_weights.get(s.conviction, 1) for s in group) / total_weight
            
            sentiment_scores[sentiment_type] = {
                'score': total_weight,
                'intensity': weighted_intensity,
                'confidence': weighted_confidence,
                'count': len(group),
                'sentiments': group
            }
        
        # Pick the sentiment type with highest conviction weight
        winning_sentiment = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k]['score'])
        winner_data = sentiment_scores[winning_sentiment]
        
        resolved = self._merge_sentiment_group(winner_data['sentiments'], "conviction_weighted")
        
        logger.debug(f"Conviction-weighted resolution for {sentiments[0].ticker}: "
                    f"{winning_sentiment} (score={winner_data['score']:.1f})")
        
        return resolved
    
    def _apply_majority_vote_resolution(self, sentiments: List['Sentiment']) -> 'Sentiment':
        """Apply majority vote to resolve conflicts"""
        
        # Count sentiment types
        sentiment_counts = Counter(s.sentiment for s in sentiments)
        winning_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Get all sentiments of the winning type
        winning_sentiments = [s for s in sentiments if s.sentiment == winning_sentiment]
        
        resolved = self._merge_sentiment_group(winning_sentiments, "majority_vote")
        
        logger.debug(f"Majority vote resolution for {sentiments[0].ticker}: "
                    f"{winning_sentiment} ({len(winning_sentiments)}/{len(sentiments)})")
        
        return resolved
    
    def _apply_highest_confidence_resolution(self, sentiments: List['Sentiment']) -> 'Sentiment':
        """Take the sentiment with highest confidence"""
        
        highest_confidence_sentiment = max(sentiments, key=lambda s: s.confidence)
        
        # Create a copy with updated explanation
        resolved = Sentiment(
            confidence=highest_confidence_sentiment.confidence,
            ticker=highest_confidence_sentiment.ticker,
            is_implicit=highest_confidence_sentiment.is_implicit,
            sentiment=highest_confidence_sentiment.sentiment,
            sentiment_intensity=highest_confidence_sentiment.sentiment_intensity,
            position=highest_confidence_sentiment.position,
            conviction=highest_confidence_sentiment.conviction,
            time_horizon=highest_confidence_sentiment.time_horizon,
            source="both",  # Likely came from multiple sources
            explanation=f"Highest confidence from {len(sentiments)} extractions (confidence={highest_confidence_sentiment.confidence}). Original: {highest_confidence_sentiment.explanation}",
            extraction_refusal=False
        )
        
        logger.debug(f"Highest confidence resolution for {sentiments[0].ticker}: "
                    f"{resolved.sentiment} (confidence={resolved.confidence})")
        
        return resolved
    
    def _apply_intensity_weighted_resolution(self, sentiments: List['Sentiment']) -> 'Sentiment':
        """Weight by sentiment_intensity scores"""
        
        # Similar to confidence weighted but use sentiment_intensity as weight
        sentiment_groups = defaultdict(list)
        for s in sentiments:
            sentiment_groups[s.sentiment].append(s)
        
        sentiment_scores = {}
        for sentiment_type, group in sentiment_groups.items():
            total_weight = sum(s.sentiment_intensity for s in group)
            weighted_confidence = sum(s.confidence * s.sentiment_intensity for s in group) / total_weight
            avg_intensity = sum(s.sentiment_intensity for s in group) / len(group)
            
            sentiment_scores[sentiment_type] = {
                'score': total_weight,
                'intensity': avg_intensity,
                'confidence': weighted_confidence,
                'count': len(group),
                'sentiments': group
            }
        
        winning_sentiment = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k]['score'])
        winner_data = sentiment_scores[winning_sentiment]
        
        resolved = self._merge_sentiment_group(winner_data['sentiments'], "intensity_weighted")
        
        return resolved
    
    def _merge_same_sentiments(self, sentiments: List['Sentiment']) -> 'Sentiment':
        """Merge sentiments that have the same sentiment value but different intensities/confidences"""
        return self._merge_sentiment_group(sentiments, "same_sentiment_merge")
    
    def _merge_sentiment_group(self, sentiments: List['Sentiment'], merge_method: str) -> 'Sentiment':
        """Merge a group of sentiments into a single sentiment"""
        
        if len(sentiments) == 1:
            return sentiments[0]
        
        # Use confidence-weighted averaging for intensity and simple average for confidence
        total_confidence_weight = sum(s.confidence for s in sentiments)
        weighted_intensity = sum(s.sentiment_intensity * s.confidence for s in sentiments) / total_confidence_weight
        avg_confidence = mean(s.confidence for s in sentiments)
        
        # Determine most common categorical fields
        most_common_position = Counter(s.position for s in sentiments).most_common(1)[0][0]
        most_common_conviction = Counter(s.conviction for s in sentiments).most_common(1)[0][0]
        most_common_time_horizon = Counter(s.time_horizon for s in sentiments).most_common(1)[0][0]
        
        # Prefer explicit tickers
        is_implicit = all(s.is_implicit for s in sentiments)
        
        # Combine sources
        sources = set(s.source for s in sentiments)
        if len(sources) > 1:
            combined_source = "both"
        else:
            combined_source = list(sources)[0]
        
        merged = Sentiment(
            confidence=round(avg_confidence),
            ticker=sentiments[0].ticker,  # All should be the same
            is_implicit=is_implicit,
            sentiment=sentiments[0].sentiment,  # All should be the same for this method
            sentiment_intensity=round(weighted_intensity),
            position=most_common_position,
            conviction=most_common_conviction,
            time_horizon=most_common_time_horizon,
            source=combined_source,
            explanation=self._merge_explanations(sentiments, merge_method),
            extraction_refusal=False
        )
        
        return merged
    
    def _merge_explanations(self, sentiments: List['Sentiment'], merge_method: str) -> str:
        """Merge explanations from multiple sentiments"""
        explanations = [s.explanation for s in sentiments if s.explanation and s.explanation.strip()]
        
        if not explanations:
            return f"Merged {len(sentiments)} sentiments using {merge_method}"
        
        if len(explanations) == 1:
            return f"{explanations[0]} (merged from {len(sentiments)} extractions using {merge_method})"
        
        # Take the longest explanation (likely most detailed) and note the merge
        longest_explanation = max(explanations, key=len)
        return f"{longest_explanation} (merged from {len(sentiments)} extractions using {merge_method})"
    
    def _passes_quality_filter(self, sentiment: 'Sentiment') -> bool:
        """Check if sentiment passes quality thresholds"""
        return (sentiment.confidence >= self.config.min_confidence_threshold and 
                sentiment.sentiment_intensity >= self.config.min_intensity_threshold and
                not (self.config.exclude_extraction_refusals and sentiment.extraction_refusal))
    
    def _apply_quality_filters(self, sentiment: 'Sentiment') -> Optional['Sentiment']:
        """Apply quality filters to a single sentiment"""
        if self._passes_quality_filter(sentiment):
            return sentiment
        else:
            self.stats['sentiments_filtered'] += 1
            return None
    
    def _determine_primary_ticker(self, extraction_results: List['SentimentResults'], 
                                 final_sentiments: List['Sentiment']) -> Optional[str]:
        """
        Determine the primary ticker for the consolidated result
        """
        if not final_sentiments:
            return None
        
        if self.config.primary_ticker_strategy == "most_frequent":
            # Count how many times each ticker appears across all extractions
            ticker_counts = Counter()
            for result in extraction_results:
                for sentiment in result.sentiments:
                    if sentiment.ticker and not sentiment.extraction_refusal:
                        normalized = sentiment.ticker.upper().replace('$', '').strip()
                        # Weight explicit tickers higher
                        weight = 1.0 if sentiment.is_implicit else 1.5
                        ticker_counts[normalized] += weight
            
            if ticker_counts:
                most_frequent = ticker_counts.most_common(1)[0][0]
                logger.debug(f"Primary ticker by frequency: {most_frequent} ({ticker_counts[most_frequent]:.1f} weighted mentions)")
                return most_frequent
        
        elif self.config.primary_ticker_strategy == "highest_confidence":
            # Take ticker with highest confidence sentiment
            highest_confidence_sentiment = max(final_sentiments, key=lambda s: s.confidence)
            logger.debug(f"Primary ticker by confidence: {highest_confidence_sentiment.ticker} "
                        f"(confidence={highest_confidence_sentiment.confidence})")
            return highest_confidence_sentiment.ticker
        
        # Fallback: first ticker in final sentiments
        return final_sentiments[0].ticker
    
    def _create_metadata(self, extraction_results: List['SentimentResults'], 
                        final_sentiments: List['Sentiment']) -> Dict[str, Any]:
        """Create metadata for the deduplicated result"""
        
        # Count original primary tickers
        original_primaries = [r.primary_ticker for r in extraction_results if r.primary_ticker]
        primary_ticker_counts = Counter(original_primaries)
        
        # Analyze sentiment distribution
        sentiment_distribution = Counter(s.sentiment for s in final_sentiments)
        conviction_distribution = Counter(s.conviction for s in final_sentiments)
        position_distribution = Counter(s.position for s in final_sentiments)
        metadata = {
            'deduplication': {
                'source_extractions': len(extraction_results),
                'original_sentiment_count': sum(len(r.sentiments) for r in extraction_results),
                'final_sentiment_count': len(final_sentiments),
                'conflicts_resolved': len(self.conflicts_resolved),
                'strategy': self.config.strategy.value,
                'quality_filters': {
                    'min_confidence': self.config.min_confidence_threshold,
                    'min_intensity': self.config.min_intensity_threshold,
                    'sentiments_filtered': self.stats['sentiments_filtered'],
                    'extraction_refusals_excluded': self.stats['extraction_refusals_excluded']
                }
            },
            'sentiment_analysis': {
                'sentiment_distribution': dict(sentiment_distribution),
                'conviction_distribution': dict(conviction_distribution),
                'position_distribution': dict(position_distribution),
                'implicit_tickers_found': self.stats['implicit_tickers_found']
            },
            'primary_ticker_analysis': {
                'strategy': self.config.primary_ticker_strategy,
                'original_primaries': dict(primary_ticker_counts),
                'primary_changes': self.stats['primary_ticker_changes']
            },
            'conflict_summary': [
                {
                    'ticker': conflict.ticker,
                    'conflict_count': len(conflict.conflicting_sentiments),
                    'resolution': conflict.resolution_method,
                    'confidence_spread': conflict.confidence_spread,
                    'conviction_levels': conflict.conviction_levels,
                    'final_sentiment': conflict.final_sentiment.sentiment,
                    'final_conviction': conflict.final_sentiment.conviction
                }
                for conflict in self.conflicts_resolved
            ]
            }
    
    def _create_empty_result(self) -> 'SentimentResults':
        """Create empty result when no extractions provided"""
        return SentimentResults(
            sentiments=[],
            primary_ticker=None,
            metadata={'deduplication': {'source_extractions': 0, 'error': 'No extraction results provided'}}
        )
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get statistics about the deduplication process"""
        return {
            'total_extractions_processed': self.stats['total_extractions_processed'],
            'unique_tickers_found': self.stats['unique_tickers_found'],
            'conflicts_resolved': self.stats['conflicts_resolved'],
            'sentiments_filtered': self.stats['sentiments_filtered'],
            'extraction_refusals_excluded': self.stats['extraction_refusals_excluded'],
            'implicit_tickers_found': self.stats['implicit_tickers_found'],
            'primary_ticker_changes': self.stats['primary_ticker_changes'],
            'current_conflicts': len(self.conflicts_resolved)
        }
    
    def get_conflict_report(self) -> List[Dict[str, Any]]:
        """Get detailed report of conflicts that were resolved"""
        return [
            {
                'ticker': conflict.ticker,
                'conflicting_sentiments': [
                    {
                        'sentiment': s.sentiment,
                        'sentiment_intensity': s.sentiment_intensity,
                        'confidence': s.confidence,
                        'conviction': s.conviction,
                        'position': s.position,
                        'is_implicit': s.is_implicit,
                        'source': s.source
                    }
                    for s in conflict.conflicting_sentiments
                ],
                'resolution_method': conflict.resolution_method,
                'final_sentiment': {
                    'sentiment': conflict.final_sentiment.sentiment,
                    'sentiment_intensity': conflict.final_sentiment.sentiment_intensity,
                    'confidence': conflict.final_sentiment.confidence,
                    'conviction': conflict.final_sentiment.conviction,
                    'position': conflict.final_sentiment.position,
                    'is_implicit': conflict.final_sentiment.is_implicit
                },
                'confidence_spread': conflict.confidence_spread,
                'conviction_levels': conflict.conviction_levels
            }
            for conflict in self.conflicts_resolved
        ]
    
    def reset_stats(self):
        """Reset internal statistics"""
        self.stats = {
            'total_extractions_processed': 0,
            'unique_tickers_found': 0,
            'conflicts_resolved': 0,
            'sentiments_filtered': 0,
            'extraction_refusals_excluded': 0,
            'implicit_tickers_found': 0,
            'primary_ticker_changes': 0
        }
        self.conflicts_resolved = []