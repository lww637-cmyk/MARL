"""
Fetches historical Brent and WTI crude oil price data
from Yahoo Finance (yfinance) and FRED API.
Saves raw data to data/raw/.
"""

import yfinance as yf
from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    # TODO: download OHLCV data for given ticker and date range
    pass

def fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    # TODO: fetch macroeconomic series from FRED using FRED_API_KEY
    pass

def fetch_all(config: dict) -> dict:
    # TODO: fetch WTI, Brent, DXY using config settings and return as dict
    pass