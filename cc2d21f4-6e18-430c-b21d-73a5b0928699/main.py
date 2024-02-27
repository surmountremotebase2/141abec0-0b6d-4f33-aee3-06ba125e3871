from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import RSI, SMA, MACD
from surmount.data import Asset
from surmount.logging import log
import numpy as np

class TradingStrategy(Strategy):

    def __init__(self):
        # Portfolio of stocks related to AI and ML industries.
        self.tickers = ["NVDA", "GOOGL", "AMZN", "MSFT", "IBM"]
        # Assume pre-built AI/ML model scores for each stock indicating bullishness.
        # In practice, incorporate dynamic AI/ML model predictions here.
        self.ai_ml_model_scores = {ticker: np.random.uniform(0.5, 1.0) for ticker in self.tickers}

    @property
    def interval(self):
        return "1day"

    @property
    def assets(self):
        # Assets are the tickers we're interested in.
        return self.tickers

    def run(self, data):
        allocation_dict = {}
        total_score = sum(self.ai_ml_model_scores.values())

        for ticker in self.tickers:
            stock_data = data["ohlcv"]
            if ticker not in stock_data:
                continue
            
            # Basic momentum strategy using RSI & MACD to validate AI/ML model suggestions.
            rsi_indicator = RSI(ticker, stock_data, 14)
            macd_indicator = MACD(ticker, stock_data, 26, 12)
            
            if rsi_indicator and macd_indicator:
                # Check for bullish indicators: RSI above 50 and MACD line above signal line.
                is_bullish = rsi_indicator[-1] > 50 and macd_indicator["MACD"][-1] > macd_indicator["signal"][-1]
                # Scale investment based on AI/ML model score, adjusted for bullish indicators.
                allocation_dict[ticker] = self.ai_ml_model_scores[ticker] / total_score if is_bullish else 0
        
        # Ensure allocations do not exceed 100% of capital.
        if sum(allocation_dict.values()) > 1:
            # Normalize allocations to ensure they sum up to 1 (100%)
            total_allocations = sum(allocation_dict.values())
            allocation_dict = {ticker: alloc / total_allocations for ticker, alloc in allocation_dict.items()}
        
        return TargetAllocation(allocation_dict)