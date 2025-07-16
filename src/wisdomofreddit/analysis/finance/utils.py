from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Sequence, Dict, Any
import seaborn as sns



def prepare_single_stock_data(ticker, date_b, date_e, consensus_storage, financial_data, 
                              window_short=7, window_long=21, use_log_transform=False):
    """
    Loads, processes, and merges sentiment and financial data for a single stock.
    Accepts EMA window sizes as arguments.
    """
    ticker_data = consensus_storage.local_storage.get_ticker_history(ticker, date_b, date_e)
    if not ticker_data:
        return None

    sentiment_data = [
        {
            'Date': s.date, 
            'net_conviction_score': (
                s.conviction_weighted_bullish_score / s.sentiment_distribution.get('bullish', 1) - s.conviction_weighted_bearish_score /  s.sentiment_distribution.get('bearish', 1)
            ),
            'total_mentions': s.total_mentions,
            'avg_confidence': s.avg_confidence,
            'avg_sentiment_intensity': s.avg_sentiment_intensity,
            'unique_submissions': s.unique_submissions
        } 
            for s in ticker_data
        ]
    
    sentiment_df = pd.DataFrame(sentiment_data)
    if sentiment_df.empty:
        return None
        
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_df.set_index('Date', inplace=True)
    
    full_date_range = pd.date_range(start=pd.to_datetime(date_b), end=pd.to_datetime(date_e), freq='D')
    sentiment_df = sentiment_df.reindex(full_date_range).fillna(0)
    
    if use_log_transform:
        sentiment_df['predictor_score'] = np.sign(sentiment_df['net_conviction_score']) * np.log1p(np.abs(sentiment_df['net_conviction_score']))
    else:
        sentiment_df['predictor_score'] = sentiment_df['net_conviction_score']

    # Use the passed-in arguments for window sizes
    sentiment_df['predictor_ema_short'] = sentiment_df['predictor_score'].ewm(span=window_short, adjust=True).mean()
    sentiment_df['predictor_ema_long'] = sentiment_df['predictor_score'].ewm(span=window_long, adjust=True).mean()
    
    f_data_slice = financial_data[financial_data.Ticker.eq(ticker)]
    if f_data_slice.empty:
        print(f"Warning: No financial data found for ticker {ticker}. Skipping.")
        return None
        
    merged_df = sentiment_df.reset_index().rename(columns={'index': 'Date'})
    merged_df = merged_df.merge(f_data_slice, on='Date', how='left')
    
    return merged_df

def analyze_and_plot_stock(ticker, date_b, date_e, consensus_storage, financial_data, 
                           window_short=7, window_long=21, horizon_days=5, use_log_transform=False, filter_non_zero_mention=False):
    """
    Generates a full suite of analysis plots for a given stock.
    Allows for custom EMA window sizes.
    """
    df = prepare_single_stock_data(
        ticker, date_b, date_e, consensus_storage, financial_data,
        window_short=window_short,      # Pass arguments down
        window_long=window_long,        # Pass arguments down
        use_log_transform=use_log_transform
    )
    
    if df is None:
        print(f"Could not generate data for {ticker}.")
        return

    print(f"--- Generating Analysis Plots for {ticker} (Windows: {window_short}/{window_long}) ---")
    
    # The rest of the plotting code remains the same...
    plot_df = df.dropna(subset=['Open', 'Close', 'High', 'Low']).copy()
    if filter_non_zero_mention:
        plot_df = plot_df.loc[plot_df['total_mentions'].ge(1), :]
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(
        go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name=f'{ticker} Price'), secondary_y=False)
    fig1.add_trace(
        go.Scatter(x=plot_df['Date'], y=plot_df['predictor_ema_short'], name='Sentiment EMA', line=dict(color='purple')), secondary_y=True)
    fig1.update_layout(title_text=f"Test 1: Sentiment Overlay for {ticker}")
    fig1.show()


    plot_df[f'forward_return_{horizon_days}d'] = (plot_df['Close'].shift(-horizon_days) - plot_df['Close']) / plot_df['Close'] * 100
    scatter_df = plot_df.dropna(subset=['predictor_ema_short', f'forward_return_{horizon_days}d']).copy()
    scatter_df['year'] = scatter_df['Date'].dt.year.astype(str)
    fig4 = px.scatter(
        scatter_df,
        x='predictor_ema_short',
        color='year',
        y=f'forward_return_{horizon_days}d',
        title=f"Test 4: Sentiment Trend vs. Next {horizon_days}-Day Forward Return (%) for {ticker}",
        labels={'predictor_ema_short': "Today's Net Conviction EMA", f'forward_return_{horizon_days}d': f"Forward {horizon_days}-Day Return (%)"},
        trendline="ols"
    )
    fig4.show()


def prepare_base_stock_data(ticker, date_b, date_e, consensus_storage, financial_data, use_log_transform=False):
    """
    Loads and prepares the base data for a stock just ONCE.
    This includes loading from storage, reindexing, and merging with financial data.
    It does NOT calculate any EMAs.
    """
    ticker_data = consensus_storage.local_storage.get_ticker_history(ticker, date_b, date_e)
    if not ticker_data:
        return None

    sentiment_data = [
        {
            'Date': s.date, 
            'net_conviction_score': (
                s.conviction_weighted_bullish_score / s.sentiment_distribution.get('bullish', 1) - s.conviction_weighted_bearish_score /  s.sentiment_distribution.get('bearish', 1)
            ),
            'total_mentions': s.total_mentions,
            'avg_confidence': s.avg_confidence,
            'avg_sentiment_intensity': s.avg_sentiment_intensity,
            'unique_submissions': s.unique_submissions
        } 
            for s in ticker_data
        ]
    
    sentiment_df = pd.DataFrame(sentiment_data)
    if sentiment_df.empty:
        return None
        
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_df.set_index('Date', inplace=True)
    
    full_date_range = pd.date_range(start=pd.to_datetime(date_b), end=pd.to_datetime(date_e), freq='D')
    sentiment_df = sentiment_df.reindex(full_date_range).fillna(0)
    
    # Create the base score for predictors
    if use_log_transform:
        sentiment_df['predictor_score'] = np.sign(sentiment_df['net_conviction_score']) * np.log1p(np.abs(sentiment_df['net_conviction_score']))
    else:
        sentiment_df['predictor_score'] = sentiment_df['net_conviction_score']
    
    f_data_slice = financial_data[financial_data.Ticker.eq(ticker)]
    if f_data_slice.empty:
        # This check is important to avoid errors on stocks with no price data
        return None
        
    merged_df = sentiment_df.reset_index().rename(columns={'index': 'Date'})
    merged_df = merged_df.merge(f_data_slice, on='Date', how='left')
    
    return merged_df


def find_momentum_signals(tickers, date_b, date_e, consensus_storage, financial_data, 
                          min_sentiment_days=50, 
                          p_value_threshold=0.1, 
                          use_log_transform=False, 
                          filter_non_zero_mention=False,
                          short_windows_to_test=None,
                          horizons_to_test=None
                          ):
    """
    Efficiently scans tickers and a range of EMA windows to find the best momentum signals.
    """
    all_significant_results = []
    if short_windows_to_test is None:
        short_windows_to_test = [3, 5, 7, 10, 15, 20]
    if horizons_to_test is None:
        horizons_to_test = [1, 2, 3, 5, 10, 15, 20]
    
    print(f"Scanning {len(tickers)} tickers for predictive momentum signals...")
    
    for i, ticker in enumerate(tickers):
        print(f"Processing ({i+1}/{len(tickers)}): {ticker}", end='\r')
        
        if len(consensus_storage.local_storage.ticker_index.get(ticker, [])) < min_sentiment_days:
            continue

        # 1. PREPARE DATA ONCE
        df = prepare_base_stock_data(ticker, date_b, date_e, consensus_storage, financial_data, use_log_transform)
        if df is None:
            continue
            
        # 2. ENGINEER ALL FEATURES AT ONCE
        # Calculate all EMA windows
        for win in short_windows_to_test:
            df[f'predictor_ema_{win}'] = df['predictor_score'].ewm(span=win, adjust=True).mean()
        
        # Calculate all forward returns
        for n_days in horizons_to_test:
            df[f'forward_return_{n_days}d'] = (df['Close'].shift(-n_days) - df['Close']) / df['Close'] * 100

        # 3. ANALYZE PRE-COMPUTED FEATURES
        for short_win in short_windows_to_test:
            predictor_col = f'predictor_ema_{short_win}'
            
            for n_days in horizons_to_test:
                outcome_col = f'forward_return_{n_days}d'
                
                subset_df = df.dropna(subset=[predictor_col, outcome_col])
                if filter_non_zero_mention:
                    subset_df = subset_df.loc[subset_df['total_mentions'].ge(1), :]
                if len(subset_df) < min_sentiment_days:
                    continue
                X = sm.add_constant(subset_df[predictor_col])
                y = subset_df[outcome_col]
                model = sm.OLS(y, X).fit()
                
                p_value = model.pvalues[predictor_col]
                coefficient = model.params[predictor_col]
                intercept = model.params['const']
                
                # Filter for significant MOMENTUM signals only
                if p_value < p_value_threshold:
                    all_significant_results.append({
                        'ticker': ticker,
                        'horizon_days': n_days,
                        'win_short': short_win,
                        'p_value': p_value,
                        'coefficient': coefficient,
                        'signal_type': "Momentum" if coefficient > 0 else "Contrarian",
                        'r_squared': model.rsquared,
                        'intercept': intercept
                    })

    print("\nScan complete.")
    if not all_significant_results:
        print("No significant momentum signals found.")
        return pd.DataFrame()

    # Find the BEST signal for each ticker
    results_df = pd.DataFrame(all_significant_results)
    best_signals = results_df.loc[results_df.groupby('ticker')['p_value'].idxmin()]
    
    return best_signals.sort_values(by='p_value').reset_index(drop=True)


def calculate_portfolio_returns(trades_df, initial_capital):
    """
    Calculate time-weighted and money-weighted returns
    """
    # Simple approach: track capital over time
    capital_curve = [initial_capital]
    
    for _, trade in trades_df.iterrows():
        # Each trade's impact on capital
        capital_after_trade = capital_curve[-1] * (1 + trade['net_return'] * trade['position_size_pct'])
        capital_curve.append(capital_after_trade)
    
    final_capital = capital_curve[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    
    return total_return, capital_curve


def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns series"""
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def print_backtest_results(results):
    """Pretty print backtest results"""
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS FOR {results['ticker']}")
    print(f"{'='*60}")
    print(f"Signal Type: {results['signal_params']['signal_type']}")
    print(f"Period: {results['backtest_period']}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"Avg Return per Trade: {results['avg_return_per_trade_pct']:.2f}%")
    print(f"Avg Days Held: {results['avg_days_held']:.1f}")
    print(f"\n--- PERFORMANCE COMPARISON ---")
    print(f"Strategy Total Return: {results['total_strategy_return_pct']:.2f}%")
    print(f"Buy & Hold Return: {results['buy_hold_return_pct']:.2f}%")
    print(f"Excess Return: {results['excess_return_pct']:.2f}%")
    print(f"\n--- RISK METRICS ---")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Annualized Return: {results['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")



def plot_equity_curve(results: dict, initial_capital: float = 1.0):
    """
    Plots the strategy equity curve against the Buy & Hold benchmark,
    both starting from `initial_capital`.

    Args:
        results: The dict returned by backtest_single_stock_strategy.
        initial_capital: Starting cash for both strategy & buy-and-hold.
    """
    if 'error' in results:
        print(f"Cannot plot equity curve due to backtest error: {results['error']}")
        return

    ticker    = results['ticker']
    price_df  = results['price_df'].copy().reset_index()  # ensure 'Date' column
    trades_df = results['trades_df'].copy()

    # 1) Build benchmark (buy-and-hold) equity in dollars
    price_df['benchmark_equity'] = (
        price_df['Close'] / price_df['Close'].iloc[0]
    ) * initial_capital

    # 2) Map strategy daily P&L onto the price index
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    # net_return is in fractional terms: e.g. 0.02 for +2%
    daily_returns = trades_df.set_index('exit_date')['net_return']
    price_df['strategy_daily_return'] = price_df['Date'].map(daily_returns).fillna(0)

    # 3) Build strategy equity in dollars
    price_df['strategy_equity'] = (
        initial_capital * (1 + price_df['strategy_daily_return']).cumprod()
    )

    # 4) Plot both curves
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(price_df['Date'],
            price_df['strategy_equity'],
            label='Strategy',
            linewidth=2)
    ax.plot(price_df['Date'],
            price_df['benchmark_equity'],
            label=f'Buy & Hold {ticker}',
            linestyle='--',
            linewidth=2)

    # Formatting
    ax.set_title(f'Equity Curve: Strategy vs. Buy & Hold for {ticker}', fontsize=16)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True)

    # 5) Annotate final stats in $ and %
    strat_final_val = price_df['strategy_equity'].iloc[-1]
    bench_final_val = price_df['benchmark_equity'].iloc[-1]
    strat_ret_pct  = (strat_final_val / initial_capital - 1) * 100
    bench_ret_pct  = (bench_final_val / initial_capital - 1) * 100
    sharpe         = results.get('sharpe_ratio', float('nan'))
    max_dd         = results.get('max_drawdown_pct', float('nan'))

    stats_text = (
        f"Initial Capital: ${initial_capital:,.0f}\n"
        f"Strategy Final: ${strat_final_val:,.0f} ({strat_ret_pct:.2f}%)\n"
        f"Buy & Hold : ${bench_final_val:,.0f} ({bench_ret_pct:.2f}%)\n"
        f"Sharpe Ratio: {sharpe:.2f}\n"
        f"Max Drawdown: {max_dd:.2f}%"
    )
    ax.text(0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='wheat', alpha=0.5))

    plt.show()


def plot_rolling_signal_diagnostics(
    ticker,
    date_b,
    date_e,
    consensus_storage,
    financial_data,
    rolling_window_days=180,  # How many past days to use for each model fit (e.g., ~6 months)
    ema_window=7,             # The EMA window for the sentiment predictor
    horizon_days=10,          # The forward return horizon to predict
    use_log_transform=False,
    filter_non_zero_mention=False,
    min_sentiment_days_in_window=30
):
    """
    Performs a walk-forward analysis by fitting a model on a rolling window of past data.
    It then plots the evolution of the model's key statistics (coefficient, p-value, R-squared) over time.
    This helps identify periods ("regimes") where the sentiment signal was predictive.
    """
    print(f"--- Generating Rolling Signal Diagnostics for {ticker} ---")
    print(f"Rolling Window: {rolling_window_days} days | EMA: {ema_window} days | Horizon: {horizon_days} days")

    # 1. Prepare the base data with all necessary columns
    base_df = prepare_base_stock_data(ticker, date_b, date_e, consensus_storage, financial_data, use_log_transform)
    if base_df is None:
        print(f"Could not prepare base data for {ticker}.")
        return

    # Engineer the specific features we need for this analysis
    predictor_col = f'predictor_ema_{ema_window}'
    outcome_col = f'forward_return_{horizon_days}d'
    
    base_df[predictor_col] = base_df['predictor_score'].ewm(span=ema_window, adjust=True).mean()
    base_df[f'forward_return_{horizon_days}d'] = (base_df['Close'].shift(-horizon_days) - base_df['Close']) / base_df['Close'] * 100
    
    # Drop rows where we can't do the analysis
    analysis_df = base_df.dropna(subset=['Close', predictor_col, outcome_col]).set_index('Date')
    
    rolling_results = []
    
    # 2. Iterate through time (weekly steps for efficiency)
    # We start after the first full rolling window is available
    for i in range(rolling_window_days, len(analysis_df), 5): # Step by 5 days (approx. 1 week)
        current_date = analysis_df.index[i]
        window_start_date = analysis_df.index[i - rolling_window_days]
        
        # Define the data slice for the current rolling window
        window_df = analysis_df.loc[window_start_date:current_date]
        if filter_non_zero_mention:
            window_df = window_df.loc[window_df['total_mentions'].ge(1), :]
        
        if len(window_df) < min_sentiment_days_in_window: # Ensure enough data points for a meaningful regression
            continue
            
        # 3. Fit a model on the data from the current window
        X = sm.add_constant(window_df[predictor_col])
        y = window_df[outcome_col]
        
        try:
            model = sm.OLS(y, X).fit()
            
            # 4. Extract and store the model's statistics
            rolling_results.append({
                'Date': current_date,
                'Coefficient': model.params[predictor_col],
                'P_Value': model.pvalues[predictor_col],
                'R_Squared': model.rsquared
            })
        except Exception as e:
            # This can happen if data in the window is pathological (e.g., all zeros)
            continue

    if not rolling_results:
        print("Could not generate rolling results. Not enough data or too many errors.")
        return

    results_df = pd.DataFrame(rolling_results)
    results_df['neg_log_P_value'] = -np.log10(results_df['P_Value'])

    # 5. Plot the evolution of the model statistics
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Model Coefficient (Slope) Over Time",
            "Model -log10(P-Value Over Time) (Higher is Better)",
            "Model R-Squared Over Time (Explanatory Power)"
        )
    )

    # Plot 1: Coefficient (Slope)
    fig.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Coefficient'], name='Slope'), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.update_yaxes(title_text="Slope", row=1, col=1)

    # Plot 2: P-Value
    fig.add_trace(go.Scatter(x=results_df['Date'], y=results_df['neg_log_P_value'], name='-log10(P-Value)'), row=2, col=1)
    fig.add_hline(y=-np.log10(0.1), line_dash="dash", line_color="red", row=2, col=1, annotation_text="Significance Threshold (0.10)")
    fig.update_yaxes(title_text="-log10(P-Value)", row=2, col=1)

    # Plot 3: R-Squared
    fig.add_trace(go.Scatter(x=results_df['Date'], y=results_df['R_Squared'], name='R-Squared'), row=3, col=1)
    fig.update_yaxes(title_text="R²", row=3, col=1)

    fig.update_layout(
        height=800,
        title_text=f"<b>Rolling Signal Diagnostics for {ticker}</b><br><sup>Walk-Forward Analysis of Sentiment's Predictive Power</sup>",
        showlegend=False
    )
    fig.show()

def calculate_position_size(capital, base_position_pct, r_squared, 
                          min_position_pct=0.1, max_position_pct=0.8,
                          r_squared_scaling_factor=2.0):
    """
    Adjust position size based on model confidence (R-squared).
    
    Parameters:
    - base_position_pct: Base position size (e.g., 0.5 = 50%)
    - r_squared: Model R-squared value
    - min_position_pct: Minimum position size (safety floor)
    - max_position_pct: Maximum position size (risk ceiling)
    - r_squared_scaling_factor: How aggressively to scale with R-squared
    """
    
    if r_squared <= 0:
        return capital * min_position_pct
    
    # Scale position size based on R-squared
    # Higher R-squared → larger position
    confidence_multiplier = 1 + (r_squared * r_squared_scaling_factor)
    adjusted_position_pct = base_position_pct * confidence_multiplier
    
    # Apply bounds
    adjusted_position_pct = np.clip(adjusted_position_pct, min_position_pct, max_position_pct)
    
    return capital * adjusted_position_pct

def run_walk_forward_backtest(
    ticker,
    date_b,
    date_e,
    consensus_storage,
    financial_data,
    # --- Parameters for the adaptive model ---
    rolling_window_days=252,
    min_training_days=45,
    re_fit_frequency_days=21,
    ema_window=10,
    horizon_days=15,
    embargo_days=5,  # NEW: The gap between training and prediction
    p_value_threshold=0.10,
    # --- Parameters for the backtest execution ---
    initial_capital=10000,
    position_size_pct=0.5,
    min_position_pct=0.1,
    max_position_pct=0.8,
    r_squared_scaling_factor=2.0,
    entry_threshold_abs=2.0,
    use_r_squared_sizing=True,
    use_log_transform=False,
    filter_non_zero_mention=False,
    min_sentiment_days_in_window=30

):
    """
    Runs a backtest using a walk-forward methodology with a proper embargo period
    to prevent lookahead bias.
    """
    print(f"--- Running Walk-Forward Backtest for {ticker} with {embargo_days}-day embargo ---")
    
    # --- 1. Prepare Base Data & Engineer Features ---
    df = prepare_base_stock_data(ticker, date_b, date_e, consensus_storage, financial_data, use_log_transform)
    if df is None or df.empty:
        return {"error": f"Could not prepare base data for {ticker}."}

    predictor_col = f'predictor_ema'
    outcome_col = f'forward_return'
    
    df[predictor_col] = df['predictor_score'].ewm(span=ema_window, adjust=False).mean()
    # This pre-calculation is efficient. We will be careful about how we use it.
    df[outcome_col] = (df['Close'].shift(-horizon_days) - df['Close']) / df['Close'] * 100
    df = df.dropna(subset=['Close', predictor_col]).set_index('Date')

    # --- 2. Walk-Forward Model Fitting with Embargo ---
    models_store = []
    df['model_id'] = -1
    model_id_counter = 0

    # Start the loop after enough data has accumulated for the first full training window.
    start_index = min_training_days + embargo_days + horizon_days
    for i in range(start_index, len(df), re_fit_frequency_days):
        # --- EMBARGO IMPLEMENTATION ---
        prediction_start_date = df.index[i]
        # The training data must end *before* the prediction period, separated by the embargo.
        train_cutoff_date = prediction_start_date - pd.Timedelta(days=embargo_days + horizon_days)
        # Define the training window based on this strict cutoff.
        rolling_start_date = train_cutoff_date - pd.Timedelta(days=rolling_window_days)
        train_start_date = max(df.index[0], rolling_start_date)
        # Ensure the calculated start date is within the dataframe's bounds.
        if train_start_date < df.index[0]: continue
        train_window_df = df.loc[train_start_date:train_cutoff_date].copy()
        # --- END EMBARGO IMPLEMENTATION ---
        if len(train_window_df) < min_training_days: continue
        # Now, we prepare the training data from this correctly isolated window.
        if filter_non_zero_mention:
            train_window_df = train_window_df.loc[train_window_df['total_mentions'].ge(1), :]   
        X = sm.add_constant(train_window_df[predictor_col])
        y = train_window_df[outcome_col]
        model_data = pd.concat([X, y], axis=1).dropna()
        if len(model_data) < min_sentiment_days_in_window: continue
        # Fit the model on the clean, non-future-leaking data.
        model = sm.OLS(model_data[outcome_col], model_data.drop(columns=outcome_col)).fit()
        models_store.append({
            'model_id': model_id_counter,
            'train_start': model_data.index[0], 
            'train_end': model_data.index[-1], # Store the actual end of training data
            'prediction_start': prediction_start_date,
            'is_significant': model.pvalues[predictor_col] < p_value_threshold,
            'coefficient': model.params[predictor_col],
            'p_value': model.pvalues[predictor_col],
            'intercept': model.params['const'],
            'r_squared': model.rsquared,
            'n_samples': model_data.shape[0]
        })
        
        # Apply this newly trained model to the upcoming prediction/test period.
        prediction_end_date = df.index[min(i + re_fit_frequency_days - 1, len(df) - 1)]
        df.loc[prediction_start_date:prediction_end_date, 'model_id'] = model_id_counter
        model_id_counter += 1

    # --- Sections 3 (Backtest Simulation) and 4 (Performance Analysis) ---
    # The rest of the function remains exactly the same, as it correctly uses
    # the 'model_id' column to apply the right model at the right time.
    
    if not models_store:
        return {"error": "Could not generate any valid models."}
    
    models_df = pd.DataFrame(models_store).set_index('model_id')
    df = df.reset_index().merge(models_df, on='model_id', how='left').set_index('Date')
    df['predicted_return'] = df['intercept'] + (df['coefficient'] * df[predictor_col])
    df.dropna(subset=['predicted_return'], inplace=True)
    df = df.reset_index()
    # return df, models_store
    # ... (The simulation and performance analysis code from your original function goes here without changes) ...
    # --- 3. Backtest Simulation Loop ---
    capital = initial_capital
    equity_curve = {df.iloc[0]['Date']: initial_capital}
    trades = []
    current_position = None

    for i in range(1, len(df)):
        date = df.loc[i, 'Date']
        prev_date = df.loc[i-1, 'Date']
        
        # Update equity curve with current position value (mark-to-market)
        if current_position:
            current_price = df.loc[i, 'Close']  # Current market price
            if current_position['position_type'] == 'long':
                current_position_value = current_price * current_position['shares']
            else:  # short position
                # For short: profit when price goes down
                unrealized_pnl = (current_position['entry_price'] - current_price) * current_position['shares']
                current_position_value = current_position['entry_value'] + unrealized_pnl
            
            current_equity = capital + current_position_value  # capital is remaining cash
        else:
            current_equity = capital  # All cash, no positions
        
        equity_curve[date] = current_equity

        # Check exit conditions
        if current_position:
            days_held = i - current_position['entry_idx']
            
            # Calculate current PnL for exit decision using SAME price as equity calculation
            current_price = df.loc[i, 'Close']
            if current_position['position_type'] == 'long':
                current_pnl = (current_price - current_position['entry_price']) * current_position['shares']
            else:
                current_pnl = (current_position['entry_price'] - current_price) * current_position['shares']
            
            # Exit conditions: time-based OR stop-loss OR take-profit
            should_exit = (days_held >= horizon_days) or (current_pnl < -current_position['entry_value'] * 0.25)  or (current_pnl > current_position['entry_value'] * 0.25)
            
            if should_exit:
                # OPTION 1: Exit at current Close (immediate exit)
                exit_price = current_price
                final_pnl = current_pnl  # Already calculated above
                
                # OPTION 2: Exit at next day's Open (more realistic)
                #if i + 1 < len(df):
                #    exit_price = df.loc[i + 1, 'Open']
                #    if current_position['position_type'] == 'long':
                #        final_pnl = (exit_price - current_position['entry_price']) * current_position['shares']
                #    else:
                #        final_pnl = (current_position['entry_price'] - exit_price) * current_position['shares']
                #else:
                #     # Last day, use current close
                #    exit_price = current_price
                #   final_pnl = current_pnl
                
                # Return all capital (original position value + PnL)
                capital += current_position['entry_value'] + final_pnl
                
                trades.append({
                    'position_type': current_position['position_type'],
                    'entry_date': current_position['entry_date'],
                    'exit_date': date,  # or next day if using next day's open
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'days_held': days_held,
                    'pnl': final_pnl,
                    'net_return': final_pnl / current_position['entry_value'],
                    'model_id_used': current_position['model_id'],
                    'position_size_pct_used': current_position['position_size_pct_used'],
                    'entry_value': current_position['entry_value'],
                    'shares': current_position['shares']
                })
                current_position = None

        mask_position = current_position is None and df.loc[i, 'is_significant']
        if filter_non_zero_mention:
            mask_position = mask_position and df.loc[i, 'total_mentions'] >= 1
        if current_position is None and df.loc[i, 'is_significant']:
            predicted_return = df.loc[i, 'predicted_return']
            r_squared = df.loc[i, 'r_squared']
            
            long_entry_thresh, short_entry_thresh = entry_threshold_abs, -entry_threshold_abs

            position_type = None
            if predicted_return > long_entry_thresh: position_type = 'long'
            elif predicted_return < short_entry_thresh: position_type = 'short'

            if position_type:
                entry_price = df.loc[i, 'Close']
                
                # NEW: R-squared adjusted position sizing
                if use_r_squared_sizing:
                    position_size_dollars = calculate_position_size(
                        capital=capital,
                        base_position_pct=position_size_pct,
                        r_squared=r_squared,
                        min_position_pct=min_position_pct,  
                        max_position_pct=max_position_pct, 
                        r_squared_scaling_factor=r_squared_scaling_factor
                    )
                else:
                    position_size_dollars = capital * position_size_pct
                
                if capital < position_size_dollars: 
                    continue
                
                shares = position_size_dollars / entry_price
                position_size_pct_used = position_size_dollars / capital
                capital -= position_size_dollars
                
                current_position = {
                    'position_type': position_type, 
                    'entry_idx': i, 
                    'entry_date': date,
                    'entry_price': entry_price, 
                    'shares': shares, 
                    'entry_value': position_size_dollars,
                    'model_id': df.loc[i, 'model_id'],
                    'r_squared_used': r_squared,  # Track for analysis
                    'position_size_pct_used': position_size_pct_used  # Track actual %
                }
 
    # --- 4. Performance Analysis ---
    if not trades: return {"error": "No trades were generated."}
    trades_df = pd.DataFrame(trades)
    equity_df = pd.Series(equity_curve).to_frame('Strategy')
    final_capital = equity_df['Strategy'].iloc[-1]
    total_strategy_return = (final_capital / initial_capital) - 1
    equity_df['Buy & Hold'] = initial_capital * (df.set_index('Date')['Close'] / df['Close'].iloc[0])
    buy_hold_return = (equity_df['Buy & Hold'].dropna().iloc[-1] / initial_capital) - 1
    daily_returns = equity_df['Strategy'].pct_change().dropna()
    if len(daily_returns) < 2: return {"error": "Not enough daily returns to calculate risk metrics."}
    max_drawdown = (equity_df['Strategy'] / equity_df['Strategy'].cummax() - 1).min()
    annualized_return = (1 + daily_returns.mean())**252 - 1
    annualized_std = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_std if annualized_std > 0 else 0

    return {
        'ticker': ticker, 'total_trades': len(trades_df), 'win_rate_pct': (trades_df['pnl'] > 0).mean() * 100,
        'total_strategy_return_pct': total_strategy_return * 100, 'buy_hold_return_pct': buy_hold_return * 100,
        'excess_return_pct': (total_strategy_return - buy_hold_return) * 100, 'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio, 'trades_df': trades_df, 'equity_curve_df': equity_df, 'sentiment_df': df, 'models_df': models_df
    }

def plot_equity_curve_v2(backtest_results):
    """
    Plots the equity curve from the results of a walk-forward backtest.
    """
    if 'error' in backtest_results or 'equity_curve_df' not in backtest_results:
        print(f"Cannot plot equity curve: {backtest_results.get('error', 'No equity data found.')}")
        return

    df = backtest_results['equity_curve_df']
    ticker = backtest_results['ticker']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], mode='lines', name='Adaptive Strategy'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Buy & Hold'], mode='lines', name=f'Buy & Hold {ticker}', line=dict(dash='dash')))

    # Create annotation text
    stats_text = (
        f"<b>Final Portfolio Value: ${df['Strategy'].iloc[-1]:,.2f}</b><br>"
        f"Strategy Return: {backtest_results['total_strategy_return_pct']:.2f}%<br>"
        f"Buy & Hold Return: {backtest_results['buy_hold_return_pct']:.2f}%<br>"
        f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}<br>"
        f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%"
    )

    fig.update_layout(
        title=f"Walk-Forward Backtest Equity Curve: Adaptive Strategy vs. Buy & Hold for {ticker}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        legend=dict(x=0.01, y=0.99),
        annotations=[
            dict(
                x=0.02, y=0.98, xref='paper', yref='paper',
                text=stats_text, showarrow=False, align='left',
                font=dict(size=12),
                bgcolor='rgba(255, 255, 224, 0.7)'
            )
        ]
    )
    fig.show()


def plot_comprehensive_dashboard(backtest_results):
    """
    Plots a comprehensive dashboard including the equity curve, ticker price, trade markers,
    sentiment analysis, and the rolling model diagnostics.
    """
    # --- 1. Data Validation and Extraction ---
    if 'error' in backtest_results:
        print(f"Cannot plot dashboard due to backtest error: {backtest_results['error']}")
        return
    if 'equity_curve_df' not in backtest_results or 'models_df' not in backtest_results:
        print("Error: Missing 'equity_curve_df' or 'models_df' in results.")
        return

    equity_df = backtest_results['equity_curve_df']
    models_df = backtest_results['models_df'].copy()
    trades_df = backtest_results.get('trades_df', pd.DataFrame())
    sentiment_df = backtest_results.get('sentiment_df', pd.DataFrame())
    ticker = backtest_results['ticker']
    
    # Calculate -log10(P-Value) for plotting, handling p-values of 0
    models_df['neg_log_P_value'] = -np.log10(models_df['p_value'].clip(lower=1e-100))

    # --- 2. Create the 5-Row Subplot Layout ---
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.35, 0.25, 0.15, 0.15, 0.10],  # Adjusted heights for 5 rows
        subplot_titles=(
            "Walk-Forward Backtest Equity Curve with Trades & Price",
            "Sentiment vs Price with Trade Decisions",
            "Model Coefficient (Slope) Over Time",
            "Model -log10(P-Value) Over Time (Higher is Better)",
            "Model R-Squared Over Time (Explanatory Power)"
        ),
        specs=[[{"secondary_y": True}],   # Row 1: Equity + Price
               [{"secondary_y": True}],   # Row 2: Sentiment + Price
               [{"secondary_y": False}],  # Row 3: Coefficient
               [{"secondary_y": False}],  # Row 4: P-Value
               [{"secondary_y": False}]]  # Row 5: R-Squared
    )

    # --- 3. Plot 1: Equity Curve (Row 1, Primary Y-axis) ---
    fig.add_trace(go.Scatter(
        x=equity_df.index, 
        y=equity_df['Strategy'], 
        mode='lines', 
        name='Adaptive Strategy',
        line=dict(color='blue', width=2)
    ), row=1, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=equity_df.index, 
        y=equity_df['Buy & Hold'], 
        mode='lines', 
        name=f'Buy & Hold {ticker}', 
        line=dict(dash='dash', color='lightblue', width=2)
    ), row=1, col=1, secondary_y=False)

    # Add Ticker Price to Row 1 (Secondary Y-axis)
    if not sentiment_df.empty and 'Close' in sentiment_df.columns:
        sentiment_df_plot = sentiment_df.copy()
        if 'Date' in sentiment_df_plot.columns:
            sentiment_df_plot['Date'] = pd.to_datetime(sentiment_df_plot['Date'])
            sentiment_df_plot = sentiment_df_plot.set_index('Date')
        else:
            if not isinstance(sentiment_df_plot.index, pd.DatetimeIndex):
                sentiment_df_plot.index = pd.to_datetime(sentiment_df_plot.index)
        
        fig.add_trace(go.Scatter(
            x=sentiment_df_plot.index,
            y=sentiment_df_plot['Close'],
            mode='lines',
            name=f'{ticker} Price (Equity)',
            line=dict(color='orange', width=1, dash='dot'),
            opacity=0.7,
            showlegend=False  # Don't show in legend since we'll show it in sentiment plot
        ), row=1, col=1, secondary_y=True)

    # --- 4. Plot 2: Sentiment vs Price (Row 2) ---
    if not sentiment_df.empty:
        # Plot sentiment EMA on primary y-axis
        if 'predictor_ema' in sentiment_df_plot.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_df_plot.index,
                y=sentiment_df_plot['predictor_ema'],
                mode='lines',
                name='Sentiment EMA',
                line=dict(color='purple', width=2),
                fill='tonexty' if 'predictor_ema' in sentiment_df_plot.columns else None,
                fillcolor='rgba(128, 0, 128, 0.1)'
            ), row=2, col=1, secondary_y=False)
        
        # Add actual sentiment values as dots
        if 'predictor_score' in sentiment_df_plot.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_df_plot.index,
                y=sentiment_df_plot['predictor_score'],
                mode='markers',
                name='Daily Sentiment',
                marker=dict(
                    color='darkviolet', 
                    size=3, 
                    opacity=0.6,
                    symbol='circle'
                ),
                hovertemplate='Daily Sentiment<br>Date: %{x}<br>Score: %{y:.3f}<extra></extra>'
            ), row=2, col=1, secondary_y=False)
        
        # Add horizontal line at sentiment = 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Plot ticker price on secondary y-axis
        fig.add_trace(go.Scatter(
            x=sentiment_df_plot.index,
            y=sentiment_df_plot['Close'],
            mode='lines',
            name=f'{ticker} Price',
            line=dict(color='orange', width=2),
            opacity=0.8
        ), row=2, col=1, secondary_y=True)

    # --- 5. Add Trade Markers to Both Row 1 and Row 2 ---
    if not trades_df.empty:
        # Convert date columns to datetime if they're not already
        if 'entry_date' in trades_df.columns:
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        if 'exit_date' in trades_df.columns:
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # Get values at trade dates for positioning markers
        entry_equity_values = []
        exit_equity_values = []
        entry_sentiment_values = []
        exit_sentiment_values = []
        entry_prices = []
        exit_prices = []
        entry_stock_prices = []  # Stock prices at entry/exit for positioning markers
        exit_stock_prices = []
        
        for _, trade in trades_df.iterrows():
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            
            # Get equity values
            if entry_date in equity_df.index:
                entry_equity_values.append(equity_df.loc[entry_date, 'Strategy'])
            else:
                nearest_entry = equity_df.index[equity_df.index <= entry_date]
                entry_equity_values.append(equity_df.loc[nearest_entry[-1], 'Strategy'] if len(nearest_entry) > 0 else equity_df['Strategy'].iloc[0])
            
            if exit_date in equity_df.index:
                exit_equity_values.append(equity_df.loc[exit_date, 'Strategy'])
            else:
                nearest_exit = equity_df.index[equity_df.index <= exit_date]
                exit_equity_values.append(equity_df.loc[nearest_exit[-1], 'Strategy'] if len(nearest_exit) > 0 else equity_df['Strategy'].iloc[-1])
            
            # Get sentiment values
            if not sentiment_df_plot.empty and 'predictor_ema' in sentiment_df_plot.columns:
                if entry_date in sentiment_df_plot.index:
                    entry_sentiment_values.append(sentiment_df_plot.loc[entry_date, 'predictor_ema'])
                else:
                    nearest_entry_sent = sentiment_df_plot.index[sentiment_df_plot.index <= entry_date]
                    entry_sentiment_values.append(sentiment_df_plot.loc[nearest_entry_sent[-1], 'predictor_ema'] if len(nearest_entry_sent) > 0 else 0)
                
                if exit_date in sentiment_df_plot.index:
                    exit_sentiment_values.append(sentiment_df_plot.loc[exit_date, 'predictor_ema'])
                else:
                    nearest_exit_sent = sentiment_df_plot.index[sentiment_df_plot.index <= exit_date]
                    exit_sentiment_values.append(sentiment_df_plot.loc[nearest_exit_sent[-1], 'predictor_ema'] if len(nearest_exit_sent) > 0 else 0)
            else:
                entry_sentiment_values.append(0)
                exit_sentiment_values.append(0)
            
            # Get trade prices (from trade data)
            entry_prices.append(trade.get('entry_price', 0))
            exit_prices.append(trade.get('exit_price', 0))
            
            # Get stock prices at trade dates (for positioning markers on price chart)
            if not sentiment_df_plot.empty and 'Close' in sentiment_df_plot.columns:
                if entry_date in sentiment_df_plot.index:
                    entry_stock_prices.append(sentiment_df_plot.loc[entry_date, 'Close'])
                else:
                    nearest_entry_price = sentiment_df_plot.index[sentiment_df_plot.index <= entry_date]
                    entry_stock_prices.append(sentiment_df_plot.loc[nearest_entry_price[-1], 'Close'] if len(nearest_entry_price) > 0 else trade.get('entry_price', 0))
                
                if exit_date in sentiment_df_plot.index:
                    exit_stock_prices.append(sentiment_df_plot.loc[exit_date, 'Close'])
                else:
                    nearest_exit_price = sentiment_df_plot.index[sentiment_df_plot.index <= exit_date]
                    exit_stock_prices.append(sentiment_df_plot.loc[nearest_exit_price[-1], 'Close'] if len(nearest_exit_price) > 0 else trade.get('exit_price', 0))
            else:
                entry_stock_prices.append(trade.get('entry_price', 0))
                exit_stock_prices.append(trade.get('exit_price', 0))
        
        # Separate long and short trades
        long_trades = trades_df[trades_df['position_type'] == 'long']
        short_trades = trades_df[trades_df['position_type'] == 'short']
        
        # Add trade markers to Row 1 (Equity Chart)
        if not long_trades.empty:
            long_entry_equity = [entry_equity_values[i] for i in long_trades.index]
            long_exit_equity = [exit_equity_values[i] for i in long_trades.index]
            long_entry_prices = [entry_prices[i] for i in long_trades.index]
            long_exit_prices = [exit_prices[i] for i in long_trades.index]
            
            # Long entries
            fig.add_trace(go.Scatter(
                x=long_trades['entry_date'], 
                y=long_entry_equity,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green', line=dict(width=1, color='darkgreen')),
                name='Long Entry',
                hovertemplate='Long Entry<br>Date: %{x}<br>Portfolio: $%{y:,.0f}<br>Entry Price: $%{customdata:.2f}<extra></extra>',
                customdata=long_entry_prices
            ), row=1, col=1, secondary_y=False)
            
            # Long exits
            fig.add_trace(go.Scatter(
                x=long_trades['exit_date'], 
                y=long_exit_equity,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='darkgreen', line=dict(width=1, color='green')),
                name='Long Exit',
                hovertemplate='Long Exit<br>Date: %{x}<br>Portfolio: $%{y:,.0f}<br>Exit Price: $%{customdata:.2f}<br>PnL: $%{text:,.0f}<extra></extra>',
                customdata=long_exit_prices,
                text=long_trades['pnl']
            ), row=1, col=1, secondary_y=False)
        
        if not short_trades.empty:
            short_entry_equity = [entry_equity_values[i] for i in short_trades.index]
            short_exit_equity = [exit_equity_values[i] for i in short_trades.index]
            short_entry_prices = [entry_prices[i] for i in short_trades.index]
            short_exit_prices = [exit_prices[i] for i in short_trades.index]
            
            # Short entries
            fig.add_trace(go.Scatter(
                x=short_trades['entry_date'], 
                y=short_entry_equity,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red', line=dict(width=1, color='darkred')),
                name='Short Entry',
                hovertemplate='Short Entry<br>Date: %{x}<br>Portfolio: $%{y:,.0f}<br>Entry Price: $%{customdata:.2f}<extra></extra>',
                customdata=short_entry_prices
            ), row=1, col=1, secondary_y=False)
            
            # Short exits
            fig.add_trace(go.Scatter(
                x=short_trades['exit_date'], 
                y=short_exit_equity,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='darkred', line=dict(width=1, color='red')),
                name='Short Exit',
                hovertemplate='Short Exit<br>Date: %{x}<br>Portfolio: $%{y:,.0f}<br>Exit Price: $%{customdata:.2f}<br>PnL: $%{text:,.0f}<extra></extra>',
                customdata=short_exit_prices,
                text=short_trades['pnl']
            ), row=1, col=1, secondary_y=False)

        # Add trade markers to Row 2 (Price Chart - Secondary Y-axis)
        if not long_trades.empty:
            long_entry_stock_prices = [entry_stock_prices[i] for i in long_trades.index]
            long_exit_stock_prices = [exit_stock_prices[i] for i in long_trades.index]
            long_entry_sentiment = [entry_sentiment_values[i] for i in long_trades.index]
            long_exit_sentiment = [exit_sentiment_values[i] for i in long_trades.index]
            
            # Long entries on price chart (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=long_trades['entry_date'], 
                y=long_entry_stock_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='darkgreen')),
                name='Long Entry (Price)',
                showlegend=False,  # Don't duplicate in legend
                hovertemplate='Long Entry<br>Date: %{x}<br>Stock Price: $%{y:.2f}<br>Sentiment: %{customdata:.3f}<extra></extra>',
                customdata=long_entry_sentiment
            ), row=2, col=1, secondary_y=True)
            
            # Long exits on price chart (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=long_trades['exit_date'], 
                y=long_exit_stock_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='darkgreen', line=dict(width=2, color='green')),
                name='Long Exit (Price)',
                showlegend=False,
                hovertemplate='Long Exit<br>Date: %{x}<br>Stock Price: $%{y:.2f}<br>Sentiment: %{customdata:.3f}<br>PnL: $%{text:,.0f}<extra></extra>',
                customdata=long_exit_sentiment,
                text=long_trades['pnl']
            ), row=2, col=1, secondary_y=True)
        
        if not short_trades.empty:
            short_entry_stock_prices = [entry_stock_prices[i] for i in short_trades.index]
            short_exit_stock_prices = [exit_stock_prices[i] for i in short_trades.index]
            short_entry_sentiment = [entry_sentiment_values[i] for i in short_trades.index]
            short_exit_sentiment = [exit_sentiment_values[i] for i in short_trades.index]
            
            # Short entries on price chart (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=short_trades['entry_date'], 
                y=short_entry_stock_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='darkred')),
                name='Short Entry (Price)',
                showlegend=False,
                hovertemplate='Short Entry<br>Date: %{x}<br>Stock Price: $%{y:.2f}<br>Sentiment: %{customdata:.3f}<extra></extra>',
                customdata=short_entry_sentiment
            ), row=2, col=1, secondary_y=True)
            
            # Short exits on price chart (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=short_trades['exit_date'], 
                y=short_exit_stock_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='darkred', line=dict(width=2, color='red')),
                name='Short Exit (Price)',
                showlegend=False,
                hovertemplate='Short Exit<br>Date: %{x}<br>Stock Price: $%{y:.2f}<br>Sentiment: %{customdata:.3f}<br>PnL: $%{text:,.0f}<extra></extra>',
                customdata=short_exit_sentiment,
                text=short_trades['pnl']
            ), row=2, col=1, secondary_y=True)

    # Update y-axes labels
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"{ticker} Price ($)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Sentiment Score", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"{ticker} Price ($)", row=2, col=1, secondary_y=True)

    # --- 6. Plot 3: Coefficient (Row 3) ---
    fig.add_trace(go.Scatter(x=models_df['prediction_start'], y=models_df['coefficient'], name='Slope', line=dict(color='royalblue')), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig.update_yaxes(title_text="Slope", row=3, col=1)

    # --- 7. Plot 4: P-Value (Row 4) ---
    fig.add_trace(go.Scatter(x=models_df['prediction_start'], y=models_df['neg_log_P_value'], name='-log10(P-Value)', line=dict(color='firebrick')), row=4, col=1)
    fig.add_hline(y=-np.log10(0.10), line_dash="dash", line_color="red", row=4, col=1)
    fig.update_yaxes(title_text="-log10(P)", row=4, col=1)

    # --- 8. Plot 5: R-Squared (Row 5) ---
    fig.add_trace(go.Scatter(x=models_df['prediction_start'], y=models_df['r_squared'], name='R-Squared', line=dict(color='mediumseagreen')), row=5, col=1)
    fig.update_yaxes(title_text="R²", row=5, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)

    # --- 9. Final Layout and Annotation ---
    trade_stats = ""
    if not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_pnl = trades_df['pnl'].mean()
        trade_stats = f"<br>Total Trades: {total_trades}<br>Win Rate: {win_rate:.1f}%<br>Avg PnL: ${avg_pnl:,.0f}"
        if 'IC' in backtest_results:
            trade_stats += f"<br>IC: {backtest_results['IC']:.2f}"
        if 'IR' in backtest_results:
            trade_stats += f"<br>IR: {backtest_results['IR']:.2f}"
    
    stats_text = (
        f"<b>Final Portfolio Value: ${equity_df['Strategy'].iloc[-1]:,.2f}</b><br>"
        f"Strategy Return: {backtest_results['total_strategy_return_pct']:.2f}%<br>"
        f"Buy & Hold Return: {backtest_results['buy_hold_return_pct']:.2f}%<br>"
        f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}<br>"
        f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%"
        f"{trade_stats}"
    )

    fig.update_layout(
        height=1100,  # Increased height for 5 rows
        title_text=f"<b>Comprehensive Backtest & Diagnostics for {ticker}</b>",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x=0.05, y=0.9, xref='paper', yref='paper',
                text=stats_text, showarrow=False, align='left',
                font=dict(size=12),
                bgcolor='rgba(255, 255, 224, 0.7)'
            )
        ]
    )
    fig.show()



def engineer_features(
    raw_df: pd.DataFrame,
    ema_windows: Sequence[int],
    horizons: Sequence[int],
) -> pd.DataFrame:
    """
    Adds 'ema_{w}' and 'fwd_{h}d' columns to the raw df *in-place* and
    forward-fills price so that (shifted) returns are well-defined
    even across week-ends or pre-IPO NaNs.
    """
    df = raw_df.copy()
    # Forward-fill OHLCV so that weekend sentiment rows get last valid price
    ohlc_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[ohlc_cols] = df[ohlc_cols].ffill()

    # ----- EMAs
    for w in ema_windows:
        df[f"ema_{w}"] = (
            df["predictor_score"]
            .ewm(span=w, adjust=True)
            .mean()
        )

    # ----- forward returns (%)
    for h in horizons:
        df[f"fwd_{h}d"] = (
            df["Close"].shift(-h) - df["Close"]
        ) / df["Close"] * 100.0

    return df



def pick_model_this_slice(
    slice_df: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    p_level: float = 0.10
) -> pd.Series | None:
    """
    Return the winning regression (row) for this refit step,
    or None if nothing beats the p-value hurdle.
    """
    if candidate_rows.empty:
        return None

    # keep only rows that clear the significance bar
    sig = candidate_rows[candidate_rows["p_value"] < p_level]
    if sig.empty:
        return None

    # example selection rule: largest absolute R²
    idx = sig["r_squared"].abs().idxmax()

    return sig.loc[idx].copy()      # <-- <-- <-- avoid SettingWithCopyWarning




def train_walk_forward_family(
    df_raw: pd.DataFrame,
    ema_windows: Sequence[int] = (5, 10, 20, 50, 200),
    horizons: Sequence[int] = (1, 5, 10, 15, 20),
    rolling_window_days: int = 252,
    min_training_days: int = 45,
    embargo_days: int = 2,
    re_fit_frequency_days: int = 21,
    decay_half_life: int = 84,
    p_value_threshold: float = 0.10,
    min_sentiment_days_in_window: int = 30,
    filter_non_zero_mention: bool = False,
) -> Dict[str, Any]:
    from statsmodels.stats.outliers_influence import OLSInfluence
    """
    Returns:
        df_out        – original df + columns:
                        model_id, active_horizon, active_span, predicted_return
        models_df     – metadata for *every* fitted model
    """
    # 3.1  Feature prep ------------------------------------------------------

    trade_mask     = df_raw["Close"].notna() & df_raw["Volume"].fillna(0).gt(0)
    first_trade_dt = df_raw.loc[trade_mask, "Date"].iloc[0]
    df             = df_raw[df_raw["Date"] >= first_trade_dt].reset_index(drop=True)
    df = engineer_features(df, ema_windows, horizons)

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)





    # 3.2  constants & containers -------------------------------------------
    lam = np.log(2) / decay_half_life
    weight_fn = lambda n: np.exp(-lam * np.arange(n)[::-1])

    models_store = []
    model_id_counter = 0
    df["model_id"] = np.nan          # will carry active model id

    # 3.3  walk-forward loop -------------------------------------------------
    first_useable = (
        min_training_days
        + embargo_days
        + max(horizons)
    )
    for i in range(first_useable, len(df), re_fit_frequency_days):
        # ─── fit one WLS per (w,h) ─────────────────────────────────────────
        candidate_rows = []
        pred_start = df.index[i]
        for h in horizons:
            if i + h >= len(df):         # not enough look-ahead; skip this horizon
                continue
            
            # ─── train slice boundaries ────────────────────────────────────────
            train_end = pred_start - timedelta(days=embargo_days + h)
            train_beg = train_end - timedelta(days=rolling_window_days)
            if train_beg < df.index[0]:
                train_beg = df.index[0]

            train_df = df.loc[train_beg:train_end].copy()
            if len(train_df) < min_training_days:
                continue
            if filter_non_zero_mention:
                train_df = train_df[train_df["total_mentions"] >= 1]



            y = train_df[f"fwd_{h}d"]
            if y.isna().all():
                continue
            for w in ema_windows:
                x = train_df[f"ema_{w}"]
                model_data = pd.concat([x, y, train_df["total_mentions"]], axis=1).dropna()
                if len(model_data) < min_sentiment_days_in_window:
                    continue

                X = sm.add_constant(model_data[f"ema_{w}"])
                y_vec = model_data[f"fwd_{h}d"]

                wts = weight_fn(len(model_data))

                # wts = wts * np.sqrt(1+model_data["total_mentions"].values)

               #res = sm.WLS(y_vec, X, weights=wts).fit(
                #    cov_type="HAC", cov_kwds={"maxlags": h}
                #)
                res = sm.OLS(y_vec, X).fit()
                infl       = OLSInfluence(res) 
                cooks_max  = infl.cooks_distance[0].max()
                share_nz   = (np.abs(x) > 1e-3).mean()
                n_unique_vals = x.nunique()
                mask = x.std(ddof=0) < 0.005 or share_nz < 0.2  or n_unique_vals < 20 or (cooks_max > 1)
                #mask = x.std(ddof=0) < 0.005 or share_nz < 0.2  or n_unique_vals < 20
                if mask:  
                #    # print(f'Skipping slice {pred_start}, {h}, {w}, {x.std(ddof=0):.2f}, {res.rsquared:.2f}, {share_nz:.2f}, {n_unique_vals}, {cooks_max}')
                    continue  # skip slice

                candidate_rows.append(
                    dict(
                        tmp_id=len(candidate_rows),
                        span=w,
                        horizon=h,
                        coefficient=res.params[f"ema_{w}"],
                        intercept=res.params["const"],
                        p_value=res.pvalues[f"ema_{w}"],
                        neg_log_P_value=-np.log10(res.pvalues[f"ema_{w}"]),
                        t_stat=res.tvalues[f"ema_{w}"],
                        r_squared=res.rsquared,
                        n=len(model_data),
                        train_start=model_data.index[0],
                        train_end=model_data.index[-1],
                        prediction_start=pred_start,
                        x_std=x.std(ddof=0),
                        n_unique_vals=x.nunique(),
                        rsquared_adj=res.rsquared_adj,
                        share_nonzero=share_nz,
                        cooks_max=cooks_max
                    )
                )

        if not candidate_rows:
            # nothing trainable this slice
            continue
        cand_df = pd.DataFrame(candidate_rows)

        # ─── choose active model for this slice ────────────────────────────
        winner = pick_model_this_slice(train_df, cand_df, p_value_threshold)
        if winner is None:
            # stay flat until next refit – no model id assigned
            continue

        # assign a persistent model_id
        winner["model_id"] = model_id_counter
        models_store.append(winner)
        model_id_counter += 1

        # ─── stamp df rows until next refit with this model ────────────────
        pred_end_idx = min(
            i + re_fit_frequency_days - 1,
            len(df) - 1,
        )
        pred_end_date = df.index[pred_end_idx]
        df.loc[pred_start:pred_end_date, "model_id"] = winner["model_id"]
        df.loc[pred_start:pred_end_date, "active_span"] = winner["span"]
        df.loc[pred_start:pred_end_date, "active_horizon"] = winner["horizon"]

        # pre-compute predicted_return for those rows
        df.loc[pred_start:pred_end_date, "predicted_return"] = (
            winner["intercept"]
            + winner["coefficient"] * df.loc[pred_start:pred_end_date, f"ema_{winner['span']}"]
        )



    # 3.4 wrap-up
    # models_df = pd.DataFrame(models_store).set_index("model_id")
    models_df = pd.DataFrame(models_store)
    return {"df": df.reset_index(), "models_df": models_df.set_index('model_id')}

def find_suspicious_fits(
    models_df: pd.DataFrame,
    p_threshold: float = 1e-20,
    t_threshold: float = 30.0,
) -> pd.DataFrame:
    """
    Return rows of models_df where the p-value is effectively zero
    *or* the |t| statistic is insanely high — classic signs of
    a degenerate regression.
    """
    mask = (models_df["p_value"] < p_threshold) | (models_df["t_stat"].abs() > t_threshold)
    suspects = models_df.loc[mask].copy()
    # add a couple of quick diagnostics
    suspects["obs_span"] = (suspects["train_end"] - suspects["train_start"]).dt.days
    return suspects.sort_values(by="prediction_start")

def plot_sentiment_period(
    df: pd.DataFrame,
    slice_meta: pd.Series,
    span: int | None = None,
):
    """
    Shows predictor EMA (or raw score) together with total_mentions
    and price so you can spot a single outlier day.
    """
    span = span or slice_meta["span"]
    xcol = f"ema_{span}"

    slc = df.loc[slice_meta["train_start"]: slice_meta["train_end"]]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(slc.index, slc[xcol], label=f"EMA{span}", color="purple")
    ax1.set_ylabel("Sentiment EMA")

    ax2 = ax1.twinx()
    ax2.bar(slc.index, slc["total_mentions"], label="Mentions", alpha=0.3, color="steelblue")
    ax2.set_ylabel("Total Mentions")

    ax1.set_title(f"Sentiment & mentions: {slice_meta['train_start'].date()} → "
                  f"{slice_meta['train_end'].date()}")
    fig.tight_layout()
    plt.show()


def plot_slice_scatter(
    df: pd.DataFrame,
    slice_meta: pd.Series,
):
    """
    df  – the full feature dataframe with index = Date
    slice_meta – one row from suspects_df
    """
    span = slice_meta["span"]
    h    = slice_meta["horizon"]
    xcol = f"ema_{span}"
    ycol = f"fwd_{h}d"

    slc = df.loc[slice_meta["train_start"]: slice_meta["train_end"], [xcol, ycol]].dropna()

    sns.regplot(x=xcol, y=ycol, data=slc, line_kws={"color": "red"})
    plt.title(f"Scatter in suspect window: EMA{span} → fwd{h}d\n"
              f"t={slice_meta.t_stat:.1f}, p={slice_meta.p_value:.1e} coef={slice_meta.coefficient:.1e}")
    plt.show()


def plot_param_heatmap(
    models_df: pd.DataFrame,
    param: str = "coefficient",
    index: str = "span",
    ticker: str = "TICKER",
) -> go.Figure:
    """
    Heat-map of model parameters over time.

    x-axis  – prediction_start dates
    y-axis  – categorical levels of `index` (e.g. EMA span, horizon, …)
    value   – `param` (coef, t_stat, r_squared, …)
    """
    if models_df.empty:
        raise ValueError("models_df is empty – nothing to plot.")

    pivot = (
        models_df
        .reset_index()
        .pivot_table(index=index,
                     columns="prediction_start",
                     values=param,
                     aggfunc="first")
        .sort_index(ascending=False)
    )

    # --- make y categorical by converting to strings ---------------------
    y_labels = pivot.index.astype(str)
    x_labels = pivot.columns

    fig = px.imshow(
        pivot.values,
        x=x_labels,
        y=y_labels,
        aspect="auto",
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
        labels=dict(color=param),
        title=f"{param} over time – {ticker}",
    )

    # ensure axis is treated categorically (safety net)
    fig.update_yaxes(type="category")

    fig.update_layout(height=400)
    return fig

def attach_realised_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'realised_return' column that matches each row's active horizon.

    realised_return[t] = ( Close[t+h] - Close[t] ) / Close[t]  * 100
    where h = active_horizon[t]  (integer days)
    """
    df = df.copy()
    realised = np.full(len(df), np.nan, dtype=float)   # pre-allocate

    # keep only rows that have a horizon value
    mask = df["active_horizon"].notna()
    if mask.sum() == 0:
        df["realised_return"] = realised
        return df

    # group by horizon value; convert to int for shift()
    for h_float, g in df.loc[mask].groupby("active_horizon"):
        h = int(h_float)                #  <-- fix: ensure integer
        fwd = (g["Close"].shift(-h) - g["Close"]) / g["Close"] * 100
        realised[g.index] = fwd.values  # align by original index

    df["realised_return"] = realised
    return df



def info_metrics(df: pd.DataFrame) -> dict:
    """
    Returns Information Coefficient (IC) and Information Ratio (IR)
    of forecast vs realised return, using all rows where both are present.
    """
    tmp = df.dropna(subset=["predicted_return", "realised_return"])
    if len(tmp) < 20:              # protect against tiny sample
        return {"IC": np.nan, "IR": np.nan}

    # correlation of time-series predictions with realised forward returns
    ic = tmp["predicted_return"].corr(tmp["realised_return"])

    # IR in correlation form
    ir = ic / np.sqrt(max(1e-12, 1 - ic**2))

    # ------------- optional: tracking-error version -------------
    # active = tmp["realised_return"] - tmp["predicted_return"]
    # ir_alt = active.mean() / active.std(ddof=0) * np.sqrt(252 / tmp["active_horizon"].mean())

    return {"IC": ic, "IR": ir}



def backtest_family_prediction(
    df: pd.DataFrame,
    models_df: pd.DataFrame,
    *,
    # ------- economics -------
    initial_capital: float = 10_000,
    commission_per_share: float = 0.005,        # IBKR tiered ~0.003–0.005
    slippage_bps: float = 2.0,                  # 0.02 % round-trip spread
    borrow_fee_bps_per_day: float = 1.0,        # 0.01 %/day short rebate
    # ------- signal gates & sizing -------
    entry_threshold_abs: float = 2.0,
    position_size_pct: float = 0.5,
    min_position_pct: float = 0.1,
    max_position_pct: float = 0.8,
    r_squared_scaling_factor: float = 2.0,
    sizing: str =  'raw',
    horizon_days_default: int = 20,
    filter_non_zero_mention: bool = False,
    allow_multiple_positions: bool = False,     # EXPERIMENTAL
) -> dict:
    """
    Back-test using pre-computed predictions and model meta-data.
    Now includes commission, slippage and borrow fees.
    """
    # ---------- 0. prep ----------
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    df['realised_vol_20d'] = (
        df['Close'].pct_change()
            .rolling(20)
            .std(ddof=0)     # in %
            .mul(100)
            .bfill()
    )

    trade_mask     = df["Close"].notna() & df["Volume"].fillna(0).gt(0)
    first_trade_dt = df.loc[trade_mask, "Date"].iloc[0]
    df             = df[df["Date"] >= first_trade_dt].reset_index(drop=True)

    meta_cols = ["p_value", "r_squared", "horizon"]
    df['model_id'] = df['model_id'].fillna(-1).astype(int)
    df = df.merge(models_df[meta_cols],
                  left_on="model_id",
                  right_index=True,
                  how="left")
    df["is_significant"] = (df["p_value"] < 0.10).fillna(False)
    if filter_non_zero_mention:
        df.loc[df["total_mentions"] < 1, "is_significant"] = False

    #df = (df.groupby("Date", as_index=False)
    #        .last()            # or .agg(…) if you want a different rule
    #        .reset_index(drop=True))

    # ---------- 1. containers ----------
    capital = initial_capital
    equity_curve = {df.iloc[0]["Date"]: initial_capital}
    trades = []
    positions = []       # list; keep <=1 unless allow_multiple_positions=True

    # ---------- 2. iterate ----------
    for i in range(1, len(df)):
        row = df.iloc[i]
        date, price = row["Date"], row["Close"]

        # A. Mark-to-market existing positions
        new_positions = []
        for pos in positions:
            # MTM value
            if pos["side"] == "long":
                pos_val = price * pos["shares"]
                daily_borrow = 0.0
            else:  # short
                pnl = (pos["entry_price"] - price) * pos["shares"]
                pos_val = pos["entry_value"] + pnl
                daily_borrow = (borrow_fee_bps_per_day / 1e4) * pos["entry_value"]

            pos["borrow_cost"] += daily_borrow
            pos["days_held"] += 1
            pos["current_val"] = pos_val

            # B. Exit test
            exit_time = pos["days_held"] >= pos["horizon"]
            exit_stop = pos_val <= 0.85 * pos["entry_value"] or pos_val >= 1.45 * pos["entry_value"]

            if exit_time or exit_stop:
                # --- transaction cost on exit ---
                slip_px = price * (1 + np.sign(1) * (slippage_bps / 2) / 1e4 *
                                   (1 if pos["side"] == "long" else -1))
                commission = commission_per_share * pos["shares"]
                exit_price = slip_px
                pnl = (exit_price - pos["entry_price"]) * pos["shares"] * (1 if pos["side"] == "long" else -1)
                pnl -= (commission + pos["borrow_cost"])

                capital += pos["entry_value"] + pnl
                trades.append(dict(
                    position_type=pos["side"],
                    entry_date=pos["entry_date"],
                    exit_date=date,
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    days_held=pos["days_held"],
                    pnl=pnl,
                    net_return=pnl / pos["entry_value"],
                    model_id_used=pos["model_id"],
                    position_size_pct_used=pos["pct_used"],
                    entry_value=pos["entry_value"],
                    shares=pos["shares"],
                    borrow_cost=pos["borrow_cost"],
                    commission_round=(commission_per_share * pos["shares"]) * 2
                ))
            else:
                new_positions.append(pos)

        positions = new_positions
        equity_curve[date] = capital + sum(p["current_val"] for p in positions)

        # C. Entry logic (skip if already occupied and multi-pos disabled)
        can_enter = row["is_significant"] and (allow_multiple_positions or not positions)
        if can_enter:
            pred   = row["predicted_return"]       # % over horizon
            sigma    = row["realised_vol_20d"]       # you pre-compute this column

            if pd.isna(sigma) or sigma == 0:
                continue



            if pred > entry_threshold_abs:
                side = "long"
            elif pred < -entry_threshold_abs:
                side = "short"
            else:
                side = None

            if side:
                # size
                r2 = row["r_squared"]
                pct = position_size_pct
                if sizing == 'rsquared':
                    pct *= min(max(r2 * r_squared_scaling_factor,
                                   min_position_pct),
                               max_position_pct)
                elif sizing == 'kelly':
                    # -------- Kelly sizing ----------------------------------------
                    kelly   = abs(pred) / (sigma**2)        # fraction of equity
                    pct     = np.clip(kelly,
                                    min_position_pct,
                                    max_position_pct)
                dollars = pct * capital
                if dollars < 1:       # trivial leftover cash guard
                    continue

                # --- execution price with slippage ---
                slip_px = price * (1 + (slippage_bps / 2) / 1e4 *
                                   (1 if side == "long" else -1))
                shares = dollars / slip_px
                commission = commission_per_share * shares
                dollars_effective = dollars - commission  # less upfront fee
                capital -= dollars_effective

                positions.append(dict(
                    side=side,
                    entry_date=date,
                    entry_price=slip_px,
                    shares=shares,
                    entry_value=dollars_effective,
                    pct_used=pct,
                    model_id=row["model_id"],
                    horizon=row.get("horizon", horizon_days_default),
                    days_held=0,
                    borrow_cost=0.0,
                    current_val=dollars_effective  # placeholder
                ))

    # ---------- 3. wrap-up ----------
    if not trades:
        return {"error": "No trades were generated."}

    equity_df = (pd.Series(equity_curve, name="Strategy")
                   .to_frame()
                   .assign(Buy_Hold=lambda x: initial_capital *
                           df.set_index("Date")["Close"] /
                           df["Close"].iloc[0]))
    equity_df.rename(columns={'Buy_Hold': 'Buy & Hold'}, inplace=True)
    daily_ret = equity_df["Strategy"].pct_change().dropna()
    final_cap = equity_df["Strategy"].iloc[-1]


    df = attach_realised_return(df)
    info = info_metrics(df)


    return dict(
        ticker=df["Ticker"].dropna().iat[0] if "Ticker" in df.columns else "TCKR",
        total_trades=len(trades),
        win_rate_pct=np.mean([t["pnl"] > 0 for t in trades]) * 100,
        total_strategy_return_pct=(final_cap / initial_capital - 1) * 100,
        buy_hold_return_pct=(equity_df["Buy & Hold"].iloc[-1] /
                             initial_capital - 1) * 100,
        excess_return_pct=((final_cap / initial_capital) -
                           (equity_df["Buy & Hold"].iloc[-1] /
                            initial_capital)) * 100,
        max_drawdown_pct=((equity_df["Strategy"] /
                           equity_df["Strategy"].cummax() - 1).min() * 100),
        sharpe_ratio=((1 + daily_ret.mean())**252 - 1) /
                     (daily_ret.std(ddof=0) * np.sqrt(252)),
        trades_df=pd.DataFrame(trades),
        equity_curve_df=equity_df,
        sentiment_df=df,
        models_df=models_df,
        IC=info["IC"],
        IR=info["IR"],
    )