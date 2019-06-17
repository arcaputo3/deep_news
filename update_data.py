# Load packages
import os
import numpy as np
import pickle
import pandas as pd
import alpaca_trade_api as tradeapi
import datetime
import time
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
np.random.seed(0)

api = tradeapi.REST(
    base_url=os.environ['APCA_API_BASE_URL'],
    key_id=os.environ['APCA_API_KEY_ID'],
    secret_key=os.environ['APCA_API_SECRET_KEY']
)

from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key=os.environ['NEWS_API_KEY'])


# Pick ETF Universe
symbols = [
    'XLF', # Financials
    'GDX', # Gold miners
    'VXX', # Volatility (Options)
    'EEM', # Emerging Markets
    'XRT', # S&P Retail
    'VTI', # Vanguard Total Stock Market
    'EWJ', # Japanese Market 
    'FXI', # China Large Cap
    'XHB', # S&P Homebuilders (Tracks real estate)
    'TLT', # 20 yr Treasury Bond
    'USO', # US Oil Fund
    'DBC', # Commodity Tracking
    'GLD', # Gold
    'SPY', # S&P 500
    'QQQ', # Nasdaq 100
    'XSW', # Computer Software
]


def update_news_data():
    """ Gets latest headlines and appends to text data. """
    print(f"Getting latest headlines for {datetime.datetime.now()}")
    top_hl = newsapi.get_top_headlines(sources='the-wall-street-journal,bloomberg,reuters,cnbc,fortune',)
    total_results = top_hl['totalResults']
    total_data = {
        'date': [],
        'title': [],
        'text': [],
    }
    for article in top_hl['articles']:
        total_data['date'].append(article['publishedAt'])
        total_data['title'].append(article['title'])
        total_data['text'].append(article['content'])
    
    # Rework index to appropriate day
    df = pd.DataFrame(total_data)
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index).tz_convert('US/Eastern').round('D').date
    df.reset_index(inplace=True)
    df['index'] = pd.to_datetime(df['index'])
    df.loc[df['index'].dt.weekday == 5, ['index']] = df.loc[df['index'].dt.weekday == 5, ['index']] + datetime.timedelta(days=2)
    df.loc[df['index'].dt.weekday == 6, ['index']] = df.loc[df['index'].dt.weekday == 6, ['index']] + datetime.timedelta(days=1)
    df.set_index('index', inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize('US/Eastern')
    
    # Combine to single column
    df['combined_text'] = df['title'] + ' ' + df['text']
    
    # Get and combine old data
    print("Combining with current news data and saving...")
    old_df = pd.read_csv('data/text_data.csv', index_col=0)
    old_df.index = pd.to_datetime(old_df.index, utc=True).tz_convert('US/Eastern')
    new_df = pd.concat([old_df, df], sort=True).sort_index()
    
    # Save and return data
    new_df[['combined_text']].to_csv('data/text_data.csv')
    print("Done!")
    # Return latest text for tokenization
    return df[['combined_text']]


def update_ticker_data():
    """ Gets latest minute ticker data and updates. """
    print("Getting current ticker data...")
    ticker_data = pd.read_csv("data/ticker_data.csv", index_col=0)
    ticker_data.index = pd.to_datetime(ticker_data.index, utc=True).tz_convert('US/Eastern')
    lookback = pd.to_datetime(ticker_data.index.max().date())
    lookforward = pd.to_datetime(datetime.datetime.now().date())
    
    min_btw = int(((lookforward - lookback) / pd.Timedelta(1, 'm')) / 5000 + 1)
    
    print("Loading latest data...")
    hist = {}
    for symbol in symbols:
        point = lookback
        hist[symbol] = api.polygon.historic_agg(
                            size="minute", 
                            symbol=symbol, 
                            _from=str(point),
                            limit=5000,
                        ).df
        for _ in range(1, min_btw + 1):
            # Gets last data
            point = hist[symbol].index[-1].tz_convert('US/Eastern')
            hist[symbol] = pd.concat(
                [
                    hist[symbol],
                    api.polygon.historic_agg(
                            size="minute", 
                            symbol=symbol, 
                            _from=str(point),
                            limit=5000
                        ).df
                ],
                 axis=0)
        # Remove duplicates and sort index
        hist[symbol] = hist[symbol][~hist[symbol].index.duplicated(keep='last')].sort_index()
        # Easier OHLCV readability
        hist[symbol].columns = [c + '_' + symbol for c in hist[symbol].columns]
    
    print("Combining with current ticker data and saving...")
    df = pd.concat(hist.values(), axis=1).fillna(method='ffill')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
    df.dropna(inplace=True)
    new_df = pd.concat([ticker_data, df], sort=True)
    new_df.index = pd.to_datetime(new_df.index, utc=True).tz_convert('US/Eastern')
    new_df = new_df[~new_df.index.duplicated(keep='last')]
    new_df.dropna(inplace=True)
    new_df.to_csv('data/ticker_data.csv')
    print("Done!")
    # Return latest data for conversion
    return df
    
    
def tokenize_text_data(data):
    """ Tokenizes text using latest data. """
    print("Loading tokenizer...")
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    print("Tokenizing new text data...")
    list_tokenized_headlines = tokenizer.texts_to_sequences(data['combined_text'])
    X = sequence.pad_sequences(list_tokenized_headlines, maxlen=100)
    X = pd.DataFrame(X, index=pd.to_datetime(data.index).tz_convert('US/Eastern'))
    
    print("Combining with current tokenized text data and saving...")
    old_X = pd.read_csv('data/tokenized_text.csv', index_col=0)
    old_X.index = pd.to_datetime(old_X.index, utc=True).tz_convert('US/Eastern')
    old_X.columns = X.columns
    new_X = pd.concat([old_X, X], sort=True).sort_index().astype('int64')
    new_X.to_csv('data/tokenized_text.csv')
    
    print("Done!")
    return new_X
    

def get_volatility(data, symbols=symbols):
    """ Gets daily average volatility during tradable hours. """
    out = {}
    for symbol in symbols:
        out[symbol] = data['close_' + symbol].between_time('09:30', '15:59').groupby(pd.Grouper(freq='D')).std().dropna()
    return pd.concat(out, axis=1) 


def get_volume(data, symbols=symbols):
    """ Gets daily total volume during tradable hours. """
    out = {}
    for symbol in symbols:
        out[symbol] = data['volume_' + symbol].between_time('09:30', '15:59').groupby(pd.Grouper(freq='D')).sum().dropna()
        
    return pd.concat(out, axis=1) 


def wrap_tick_data(data, symbols=symbols):
    """ Gets latest data denoted as in-play, i.e. high volume and volatility, and wraps morning tick data (8am until 10am) 
        for each of our 16 etfs as a single vector. 16 assets, 5 bars each, 121 minutes = 9680 total values per day."""
    
    # Get time horizon
    lookback = pd.to_datetime(data.index.min().date())
    lookforward = pd.to_datetime(data.index.max().date())
    
    # Get volume and volatility
    print("Getting target data...")
    volatility = get_volatility(data)
    volume = get_volume(data)
    in_play_df = ((volatility > volatility.quantile(0.5)) & (volume > volume.quantile(0.5))).dropna() 
    in_play_df = in_play_df.astype('int64')
    
    btw_days = (lookforward - lookback).days
    print("Wrapping latest tick data...")
    # Store each range
    chunks = []
    idx = []
    for day in range(btw_days + 1):
        curr_date = str((lookback + datetime.timedelta(days=day)).date())
        # Lookback over past 120 minutes
        try:
            chunks.append(data.loc[curr_date].between_time('08:00:00', '10:00:00').values.reshape(121*80,))
            idx.append(pd.to_datetime(curr_date).tz_localize('US/Eastern'))
        except:
            #print(curr_date)
            continue
    chunks = np.stack(chunks)
    data_chunks = pd.DataFrame(chunks, index=idx).join(in_play_df, how='left')
    X, y = data_chunks[[c for c in data_chunks.columns if c not in symbols]], data_chunks[symbols].fillna(0).astype('int64')
    
    print("Combining and saving wrapped tick data...")
    old_X = pd.read_csv('data/wrapped_tick_data.csv', index_col=0)
    old_X.index = pd.to_datetime(old_X.index, utc=True).tz_convert('US/Eastern')
    X.columns = old_X.columns
    new_X = pd.concat([old_X, X], sort=True).sort_index()
    new_X.to_csv('data/wrapped_tick_data.csv')
    
    print("Combining and saving volume_volatility data...")
    old_y = pd.read_csv('data/vv_target.csv', index_col=0)
    old_y.index = pd.to_datetime(old_y.index, utc=True).tz_convert('US/Eastern')
    old_y.columns = y.columns
    new_y = pd.concat([old_y, y], sort=True).sort_index()
    new_y.to_csv('data/vv_target.csv')
          
    print("Done!")
    return X, y
    
    
if __name__ == "__main__":
    """ Run a full update and time it. """
    start = time.time()
    latest_text = update_news_data()
    latest_tick = update_ticker_data()
    _ = tokenize_text_data(latest_text)
    _ = wrap_tick_data(latest_tick)
    end = time.time()
    print(f"Process took {(end - start)/60} minutes.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    