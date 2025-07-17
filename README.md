# WisdomOfReddit

This repository contains 8 years of processed stock data from Reddit and a few ways to interact with it.  
More detailed description on Medium: [link](https://medium.com/@sergeykolchenko/reddit-and-stock-returns-is-the-crowd-worth-listening-to-25b5b2dc86a3)

## Installation
`pip install -e ./`
## Usage
### 1. Getting financial data ans saving it
```python
python scripts/save_financial_data.py --fmp-api-key %FINANCIALMODELLINGPREP_API_KEY% --start-date 2017-01-01 --batch-size 250 --no-progress-save --output data/financial_data.parquet
```
### 2. Dowloading processed consensus data, 2017 to 2025
Dowload from https://drive.google.com/file/d/1IhaHg2hhdufKtGsyLlT2F2gCtYk7xIwc/view?usp=sharing and unpack it 

### 3. Downloading Reddit dump (if you want to run sentiment analysis on your own)
Download from [here](https://www.reddit.com/r/pushshift/comments/1itme1k/separate_dump_files_for_the_top_40k_subreddits/), unpack, convert to .csv with `zst_convert.py` and run `parse_dump.py`

### Build models & backtest
After steps (1) and (2) you can run `notebooks/01-plot_and_explore_stock.ipynb` to plot sentiment across different stocks