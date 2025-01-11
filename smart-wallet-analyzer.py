#before you run this script you should install dependent environment
#pip install aiohttp aioredis pandas loguru ujson numpy



import asyncio
import aiohttp
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Set
from loguru import logger
import ujson
from concurrent.futures import ThreadPoolExecutor
import aioredis
from functools import lru_cache
import numpy as np

# --- Data Models ---
@dataclass
class TradeInfo:
    token_address: str
    timestamp: datetime
    amount: Decimal
    price: Decimal
    profit: Decimal
    is_buy: bool

@dataclass
class WalletMetrics:
    address: str
    sol_balance: Decimal
    total_profit: Decimal
    profit_7d: Decimal
    profit_30d: Decimal
    win_rate_30d: float
    trade_count_30d: int
    unique_tokens_30d: int
    avg_profit_per_trade: Decimal
    score: float = 0.0
    status: str = 'active'
    delete_reason: Optional[str] = None

class SmartCache:
    def __init__(self, redis_url: str = None):
        self.redis = aioredis.from_url(redis_url) if redis_url else None
        self._local_cache: Dict = {}
        self.ttl = 24 * 3600  # 24 hours

    async def get(self, key: str) -> Optional[dict]:
        # Try local cache first
        if key in self._local_cache:
            data, expire_time = self._local_cache[key]
            if datetime.now() < expire_time:
                return ujson.loads(data)
            del self._local_cache[key]

        # Try Redis if available
        if self.redis:
            data = await self.redis.get(key)
            if data:
                return ujson.loads(data)
        return None

    async def set(self, key: str, value: dict):
        data = ujson.dumps(value)
        expire_time = datetime.now() + timedelta(seconds=self.ttl)
        
        # Update local cache
        self._local_cache[key] = (data, expire_time)
        
        # Update Redis if available
        if self.redis:
            await self.redis.set(key, data, ex=self.ttl)

class WalletAnalyzer:
    def __init__(self, api_key: str, redis_url: str = None):
        self.api_key = api_key
        self.cache = SmartCache(redis_url)
        self.session = None
        self.logger = logger.bind(service="wallet_analyzer")
        
        # Configure logging
        logger.add(
            "wallet_analyzer.log",
            rotation="100 MB",
            compression="zip",
            retention="7 days"
        )

    async def setup(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                json_serialize=ujson.dumps,
                headers={'Authorization': self.api_key}
            )

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_wallet_metrics(self, address: str) -> Optional[WalletMetrics]:
        try:
            # Check cache first
            cache_key = f"wallet_metrics:{address}"
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                return WalletMetrics(**cached_data)

            # Fetch fresh data
            await self.setup()
            async with self.session.get(f"/api/wallet/{address}/metrics") as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(loads=ujson.loads)
                
            metrics = WalletMetrics(**data)
            await self.cache.set(cache_key, data)
            return metrics

        except Exception as e:
            self.logger.error(f"Error fetching metrics for {address}: {str(e)}")
            return None

    async def get_wallet_trades(self, address: str, days: int = 30) -> List[TradeInfo]:
        try:
            cache_key = f"wallet_trades:{address}:{days}"
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                return [TradeInfo(**trade) for trade in cached_data]

            await self.setup()
            async with self.session.get(
                f"/api/wallet/{address}/trades",
                params={'days': days}
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(loads=ujson.loads)

            trades = [TradeInfo(**trade) for trade in data]
            await self.cache.set(cache_key, [vars(trade) for trade in trades])
            return trades

        except Exception as e:
            self.logger.error(f"Error fetching trades for {address}: {str(e)}")
            return []

    async def analyze_wallet(self, address: str) -> Optional[WalletMetrics]:
        """Main wallet analysis function"""
        try:
            # Get basic metrics
            metrics = await self.get_wallet_metrics(address)
            if not metrics:
                return None

            # Apply exclusion rules
            if await self._should_exclude(metrics):
                return None

            # Get recent trades
            trades = await self.get_wallet_trades(address)
            
            # Check for harvest trading
            if await self._detect_harvest_trading(trades):
                metrics.status = 'delete'
                metrics.delete_reason = 'Harvest trading detected'
                return metrics

            # Calculate score
            metrics.score = await self._calculate_score(metrics, trades)
            return metrics

        except Exception as e:
            self.logger.error(f"Error analyzing wallet {address}: {str(e)}")
            return None

    async def _should_exclude(self, metrics: WalletMetrics) -> bool:
        """Fast pre-filtering of wallets"""
        if metrics.sol_balance < 1:
            return True
        if metrics.total_profit < 0:
            return True
        if metrics.profit_7d < 15000:
            return True
        if metrics.trade_count_30d > 3000:
            return True
        if metrics.unique_tokens_30d < 10:
            return True
        return False

    async def _detect_harvest_trading(self, trades: List[TradeInfo]) -> bool:
        """Optimized harvest trading detection"""
        if not trades:
            return False

        # Convert to numpy arrays for faster processing
        timestamps = np.array([t.timestamp.timestamp() for t in trades])
        is_buy = np.array([t.is_buy for t in trades])
        amounts = np.array([float(t.amount) for t in trades])
        
        # Find consecutive buy-sell pairs within 60 seconds
        for i in range(len(trades) - 1):
            if (is_buy[i] and not is_buy[i + 1] and  # Buy followed by sell
                timestamps[i + 1] - timestamps[i] <= 60 and  # Within 60 seconds
                abs(amounts[i + 1] - amounts[i]) / amounts[i] <= 0.05):  # Similar amounts
                return True
        return False

    async def _calculate_score(self, metrics: WalletMetrics, trades: List[TradeInfo]) -> float:
        """Calculate comprehensive wallet score"""
        score = 0.0
        
        # Win rate score (0-1 points)
        score += min(1.0, metrics.win_rate_30d)
        
        # 7-day profit score (0-1 points)
        score += min(1.0, float(metrics.profit_7d) / 50000)
        
        # 30-day profit score (0-1 points)
        score += min(1.0, float(metrics.profit_30d) / 100000)
        
        # Profit stability score (0-0.5 points)
        if metrics.profit_30d > 0:
            monthly_ratio = float(metrics.profit_7d * 4 / metrics.profit_30d)
            if monthly_ratio >= 2:
                score += 0.5
            elif monthly_ratio >= 1.5:
                score += 0.25
        
        # Trading efficiency score (0-1 point)
        if trades:
            profitable_trades = sum(1 for t in trades if t.profit > 0)
            efficiency = profitable_trades / len(trades)
            score += min(1.0, efficiency)
        
        return min(7.5, score)  # Cap at 7.5 points

class BatchWalletAnalyzer:
    def __init__(self, api_key: str, redis_url: str = None, batch_size: int = 50):
        self.analyzer = WalletAnalyzer(api_key, redis_url)
        self.batch_size = batch_size

    async def analyze_wallets(self, addresses: List[str]) -> Dict[str, WalletMetrics]:
        """Analyze multiple wallets in parallel batches"""
        results = {}
        
        # Process in batches
        for i in range(0, len(addresses), self.batch_size):
            batch = addresses[i:i + self.batch_size]
            batch_tasks = [self.analyzer.analyze_wallet(addr) for addr in batch]
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Store valid results
            for addr, result in zip(batch, batch_results):
                if result:
                    results[addr] = result

        return results

    async def close(self):
        await self.analyzer.close()

async def main():
    # Configuration
    API_KEY = "your_api_key"
    REDIS_URL = "redis://localhost"
    
    # Initialize analyzer
    batch_analyzer = BatchWalletAnalyzer(API_KEY, REDIS_URL)
    
    try:
        # Example addresses (replace with your actual addresses)
        addresses = [
            "wallet1",
            "wallet2",
            # ... more addresses ...
        ]
        
        # Analyze wallets
        results = await batch_analyzer.analyze_wallets(addresses)
        
        # Process results
        df = pd.DataFrame([
            {
                'address': addr,
                'status': metrics.status,
                'score': metrics.score,
                'total_profit': float(metrics.total_profit),
                'profit_7d': float(metrics.profit_7d),
                'profit_30d': float(metrics.profit_30d),
                'win_rate': metrics.win_rate_30d,
                'delete_reason': metrics.delete_reason
            }
            for addr, metrics in results.items()
        ])
        
        # Save results
        df.to_csv('wallet_analysis_results.csv', index=False)
        
    finally:
        await batch_analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())