"""
A-Insight Pro
Event & Alternative Factor Mining Engine V1.0

Factor Categories:
  1. News Sentiment — East Money stock news NLP sentiment
  2. Search Heat — Baidu search popularity index
  3. Social Heat — Xueqiu follows/tweets, stock hot rank
  4. Fund Flow — North-bound, margin trading, individual flow
  5. Auto Discovery — Plug-and-play factor registry with auto IC testing

Design:
  - Modular: each factor category is a self-contained class
  - Auto-discovery: new data sources auto-tested against future returns
  - Graceful fallback: API failures → NaN → excluded by IC engine
  - Cache: avoids repeated API calls within same day
"""

import os, sys, json, time, hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache

# Lazy imports (only when category is enabled)
SnowNLP = None  # Will be imported on demand

REPORT_DIR = "reports"
CACHE_DIR = os.path.join(REPORT_DIR, "event_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# Utility
# ============================================================

def _cache_key(prefix, *args):
    """Generate deterministic cache key."""
    raw = f"{prefix}_{'_'.join(str(a) for a in args)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_get(key):
    """Read cached data if fresh (< 6 hours)."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > 21600:  # 6 hours
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_set(key, data):
    """Write data to cache."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        # Convert numpy types
        clean = json.loads(json.dumps(data, default=str))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False)
    except Exception:
        pass


# Financial sentiment keyword adjustments
# SnowNLP was trained on product reviews — needs calibration for financial text
FINANCIAL_BULLISH = [
    "增持", "买入", "利好", "增长", "盈利", "分红", "回购", "涨停",
    "创新高", "超预期", "扭亏", "业绩预增", "中标", "签约", "突破",
    "升级", "看好", "推荐", "加仓", "放量", "突破", "龙头",
]

FINANCIAL_BEARISH = [
    "减持", "卖出", "利空", "亏损", "暴跌", "跌停", "退市", "警示",
    "立案", "调查", "处罚", "违规", "诉讼", "债务", "违约",
    "下滑", "预亏", "风险", "警示", "问询", "监管", "停牌",
    "爆雷", "造假", "ST", "*ST", "披星", "戴帽",
]

def _adjust_financial_sentiment(text, raw_score):
    """
    Adjust SnowNLP score for financial domain.
    SnowNLP misinterprets financial keywords (e.g., '减'价=good → '减'持=bad).
    """
    text_lower = str(text).lower()
    adjustment = 0.0

    for word in FINANCIAL_BULLISH:
        if word in text_lower:
            adjustment += 0.08

    for word in FINANCIAL_BEARISH:
        if word in text_lower:
            adjustment -= 0.10

    # Clamp adjustment
    adjustment = max(-0.3, min(0.3, adjustment))
    score = raw_score + adjustment
    return max(0.0, min(1.0, score))


def _ensure_snownlp():
    """Lazy-import SnowNLP."""
    global SnowNLP
    if SnowNLP is None:
        try:
            from snownlp import SnowNLP as SN
            SnowNLP = SN
        except ImportError:
            print("  [WARN] SnowNLP not installed. Install: pip install snownlp")
            SnowNLP = False
    return SnowNLP


# ============================================================
# Category 1: News Sentiment (新闻情绪因子)
# ============================================================

class NewsSentimentFactors:
    """
    Fetch stock news from East Money, compute NLP sentiment scores.

    Factors:
      news_sentiment_mean  — avg sentiment of recent news (0=neg, 1=pos)
      news_sentiment_std   — sentiment volatility
      news_volume_5d       — number of news articles in last 5 days
      news_pos_ratio       — fraction of positive articles (>0.6)
      news_neg_ratio       — fraction of negative articles (<0.4)
    """

    def __init__(self):
        self.cache = {}

    def fetch_news(self, code, max_news=50):
        """Fetch news headlines for a stock from East Money."""
        ck = _cache_key("news", code)
        cached = _cache_get(ck)
        if cached:
            return cached

        try:
            import akshare as ak
            # Convert to East Money format (e.g., 600519 → 600519)
            symbol = str(code).zfill(6)
            df = ak.stock_news_em(symbol=symbol)
            if df is None or df.empty:
                return []
            # Extract headlines + time
            news = []
            for _, row in df.head(max_news).iterrows():
                title = str(row.get("标题", row.get("title", "")))
                pub_time = str(row.get("发布时间", row.get("pub_time", "")))
                if title:
                    news.append({"title": title, "time": pub_time})
            _cache_set(ck, news)
            return news
        except Exception as e:
            return []

    def compute_sentiment(self, text):
        """Compute sentiment score for a single text with financial adjustment."""
        sn = _ensure_snownlp()
        if not sn:
            return 0.5  # Neutral fallback
        try:
            raw = sn(text).sentiments  # 0~1
            return _adjust_financial_sentiment(text, raw)
        except Exception:
            return 0.5

    def compute_all(self, code):
        """
        Compute all news sentiment factors for a stock.
        Returns dict of factor_name → value.
        """
        news = self.fetch_news(code)
        if not news:
            return {
                "news_sentiment_mean": np.nan,
                "news_sentiment_std": np.nan,
                "news_volume_5d": 0,
                "news_pos_ratio": np.nan,
                "news_neg_ratio": np.nan,
            }

        sentiments = []
        recent_count = 0
        cutoff = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        for item in news:
            score = self.compute_sentiment(item["title"])
            sentiments.append(score)
            if item["time"] >= cutoff:
                recent_count += 1

        arr = np.array(sentiments)
        pos = (arr > 0.6).mean()
        neg = (arr < 0.4).mean()

        return {
            "news_sentiment_mean": round(float(arr.mean()), 4) if len(arr) > 0 else np.nan,
            "news_sentiment_std": round(float(arr.std()), 4) if len(arr) > 1 else np.nan,
            "news_volume_5d": recent_count,
            "news_pos_ratio": round(float(pos), 4),
            "news_neg_ratio": round(float(neg), 4),
        }


# ============================================================
# Category 2: Search Heat (搜索热度因子)
# ============================================================

class SearchHeatFactors:
    """
    Baidu search popularity + stock keyword hotness.

    Factors:
      search_heat_rank   — Baidu search ranking (lower = hotter)
      hot_rank_em        — East Money hot rank position
      hot_heat_value     — East Money heat score
    """

    def fetch_baidu_search(self, keyword):
        """Fetch Baidu search index for keyword."""
        ck = _cache_key("baidu", keyword)
        cached = _cache_get(ck)
        if cached:
            return cached

        try:
            import akshare as ak
            df = ak.stock_hot_search_baidu(symbol=keyword, date="20250728")
            if df is not None and not df.empty:
                result = int(df.iloc[0].get("rank", 999)) if "rank" in df.columns else 999
                _cache_set(ck, result)
                return result
        except Exception:
            try:
                # Try alternative: hot rank from East Money
                import akshare as ak
                df = ak.stock_hot_rank_em()
                if df is not None and not df.empty:
                    match = df[df["代码"].astype(str).str.contains(keyword[:3])]
                    if not match.empty:
                        return int(match.iloc[0].get("排名", 999))
            except Exception:
                pass
        return 999

    def fetch_hot_rank(self, code):
        """Fetch East Money hot rank detail for a stock."""
        ck = _cache_key("hotrank", code)
        cached = _cache_get(ck)
        if cached:
            return cached

        try:
            import akshare as ak
            df = ak.stock_hot_rank_detail_em(symbol=str(code).zfill(6))
            if df is not None and not df.empty:
                last = df.iloc[-1]
                result = {
                    "rank": int(last.get("排名", 999)),
                    "heat": float(last.get("热度", 0)),
                }
                _cache_set(ck, result)
                return result
        except Exception:
            pass
        return {"rank": 999, "heat": 0}

    def fetch_xueqiu_hot(self, code):
        """Fetch Xueqiu (Snowball) hot tweets and followers."""
        ck = _cache_key("xq", code)
        cached = _cache_get(ck)
        if cached:
            return cached

        result = {"followers": 0, "tweets_7d": 0}
        try:
            import akshare as ak
            symbol_map = {"6": "SH", "5": "SH", "0": "SZ", "3": "SZ", "1": "SZ"}
            prefix = str(code)[0]
            market = symbol_map.get(prefix, "SZ")
            symbol = f"{market}{str(code).zfill(6)}"

            # Follow count
            try:
                follow_df = ak.stock_hot_follow_xq(symbol=symbol)
                if follow_df is not None and not follow_df.empty:
                    result["followers"] = int(follow_df.iloc[-1].get("关注", 0))
            except Exception:
                pass

            # Hot tweets
            try:
                tweet_df = ak.stock_hot_tweet_xq(symbol=symbol)
                if tweet_df is not None and not tweet_df.empty:
                    result["tweets_7d"] = len(tweet_df)
            except Exception:
                pass

            _cache_set(ck, result)
        except Exception:
            pass
        return result

    def compute_all(self, code, name=""):
        """Compute all search/heat factors for a stock."""
        keyword = name if name else str(code).zfill(6)

        baidu_rank = self.fetch_baidu_search(keyword)
        hot = self.fetch_hot_rank(code)
        xq = self.fetch_xueqiu_hot(code)

        return {
            "search_heat_rank": float(baidu_rank) if baidu_rank < 999 else np.nan,
            "hot_rank_em": float(hot["rank"]) if hot["rank"] < 999 else np.nan,
            "hot_heat_value": float(hot["heat"]),
            "xq_followers": float(xq["followers"]) if xq["followers"] > 0 else np.nan,
            "xq_tweets_7d": float(xq["tweets_7d"]),
        }


# ============================================================
# Category 3: Fund Flow (资金流向因子)
# ============================================================

class FundFlowFactors:
    """
    Capital flow indicators.

    Factors:
      fund_flow_net         — net individual fund flow (万元)
      fund_flow_main_pct    — main fund net / total volume
      fund_flow_big_deal    — large deal proportion
      margin_balance_change — margin balance daily change
    """

    def fetch_individual_flow(self, code):
        """Fetch individual stock fund flow from East Money."""
        ck = _cache_key("flow", code)
        cached = _cache_get(ck)
        if cached:
            return cached

        try:
            import akshare as ak
            market_map = {"6": "sh", "5": "sh", "0": "sz", "3": "sz", "1": "sz"}
            prefix = str(code)[0]
            market = market_map.get(prefix, "sz")
            symbol = str(code).zfill(6)

            df = ak.stock_individual_fund_flow(stock=symbol, market=market)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                result = {
                    "net_flow": float(last.get("净流入", last.get("主力净流入", 0))),
                    "main_net": float(last.get("主力净流入", 0)),
                    "date": str(last.get("日期", "")),
                }
                _cache_set(ck, result)
                return result
        except Exception:
            pass
        return {"net_flow": 0, "main_net": 0, "date": ""}

    def fetch_north_bound_summary(self):
        """Fetch north-bound capital flow summary (market-level)."""
        ck = _cache_key("north", "summary")
        cached = _cache_get(ck)
        if cached:
            return cached

        try:
            import akshare as ak
            df = ak.stock_hsgt_fund_flow_summary_em()
            if df is not None and not df.empty:
                result = {
                    "north_net": float(df.iloc[-1].get("当日净流入", 0)),
                    "north_cum": float(df.iloc[-1].get("历史累计净流入", 0)),
                }
                _cache_set(ck, result)
                return result
        except Exception:
            pass
        return {"north_net": 0, "north_cum": 0}

    def compute_all(self, code):
        """Compute fund flow factors for a stock."""
        flow = self.fetch_individual_flow(code)
        north = self.fetch_north_bound_summary()

        return {
            "fund_flow_main_net": round(float(flow["main_net"]), 2),
            "north_bound_daily": round(float(north["north_net"]), 2),
        }


# ============================================================
# Category 4: Auto-Discovery Framework (自动因子发现)
# ============================================================

class FactorRegistry:
    """
    Plug-and-play factor registry for automatic discovery.

    Any new data source can register itself via @register_factor.
    The engine will automatically test IC against future returns.
    """

    def __init__(self):
        self._registry = {}  # name → {fn, category, description, params}

    def register(self, name, category, description, compute_fn, params=None):
        """
        Register a new factor for auto-discovery.

        Args:
            name: Factor name (e.g., 'my_new_factor')
            category: Factor category label
            description: Human-readable description
            compute_fn: callable(code, **params) → float or dict
            params: Optional default params
        """
        self._registry[name] = {
            "name": name,
            "category": category,
            "description": description,
            "fn": compute_fn,
            "params": params or {},
        }
        return self

    def list_factors(self):
        """Return all registered factors."""
        return list(self._registry.values())

    def get_factor(self, name):
        """Get a single factor by name."""
        return self._registry.get(name)

    def compute_factor(self, name, code, **kwargs):
        """Compute a single factor value for a stock."""
        entry = self._registry.get(name)
        if entry is None:
            return np.nan
        try:
            params = {**entry["params"], **kwargs}
            result = entry["fn"](code, **params)
            return result
        except Exception:
            return np.nan

    def compute_all_factors(self, code, **kwargs):
        """Compute all registered factors for a stock. Returns dict."""
        results = {}
        for name, entry in self._registry.items():
            val = self.compute_factor(name, code, **kwargs)
            if isinstance(val, dict):
                results.update(val)
            else:
                results[name] = val
        return results


# Global registry instance
factor_registry = FactorRegistry()


# ============================================================
# Register Built-in Alternative Factors
# ============================================================

def _register_builtin_factors():
    """Register all built-in alternative factors into the auto-discovery registry."""

    # --- News Sentiment Factors ---
    news_engine = NewsSentimentFactors()

    def _compute_news_sentiment_mean(code, **kw):
        return news_engine.compute_all(code).get("news_sentiment_mean", np.nan)

    def _compute_news_volume_5d(code, **kw):
        return news_engine.compute_all(code).get("news_volume_5d", 0)

    def _compute_news_pos_ratio(code, **kw):
        return news_engine.compute_all(code).get("news_pos_ratio", np.nan)

    factor_registry.register(
        "news_sentiment_mean", "news_sentiment",
        "Avg NLP sentiment of recent stock news (0=neg, 1=pos)",
        _compute_news_sentiment_mean
    )
    factor_registry.register(
        "news_volume_5d", "news_sentiment",
        "Number of news articles about stock in last 5 days",
        _compute_news_volume_5d
    )
    factor_registry.register(
        "news_pos_ratio", "news_sentiment",
        "Fraction of news with sentiment > 0.6",
        _compute_news_pos_ratio
    )

    # --- Search & Social Heat Factors ---
    heat_engine = SearchHeatFactors()

    def _compute_hot_heat_value(code, **kw):
        return heat_engine.compute_all(code).get("hot_heat_value", np.nan)

    def _compute_xq_followers(code, **kw):
        return heat_engine.compute_all(code).get("xq_followers", np.nan)

    def _compute_hot_rank_em(code, **kw):
        return heat_engine.compute_all(code).get("hot_rank_em", np.nan)

    def _compute_xq_tweets_7d(code, **kw):
        return heat_engine.compute_all(code).get("xq_tweets_7d", 0)

    factor_registry.register(
        "hot_heat_value", "social_heat",
        "East Money stock heat score (higher = hotter)",
        _compute_hot_heat_value
    )
    factor_registry.register(
        "xq_followers", "social_heat",
        "Xueqiu (Snowball) follower count",
        _compute_xq_followers
    )
    factor_registry.register(
        "hot_rank_em", "social_heat",
        "East Money hot rank position (lower = hotter)",
        _compute_hot_rank_em
    )
    factor_registry.register(
        "xq_tweets_7d", "social_heat",
        "Xueqiu hot tweet count (7 days)",
        _compute_xq_tweets_7d
    )

    # --- Fund Flow Factors ---
    flow_engine = FundFlowFactors()

    def _compute_fund_flow_main_net(code, **kw):
        return flow_engine.compute_all(code).get("fund_flow_main_net", np.nan)

    def _compute_north_bound_daily(code, **kw):
        return flow_engine.compute_all(code).get("north_bound_daily", np.nan)

    factor_registry.register(
        "fund_flow_main_net", "fund_flow",
        "Daily net main fund flow into stock (万元)",
        _compute_fund_flow_main_net
    )
    factor_registry.register(
        "north_bound_daily", "fund_flow",
        "Daily north-bound capital net flow (market-level, 万元)",
        _compute_north_bound_daily
    )


# Auto-register on import
_register_builtin_factors()


# ============================================================
# Batch Computation
# ============================================================

def compute_event_factors_for_stock(code, categories=None):
    """
    Compute all event/alternative factors for a single stock.
    Args:
        code: 6-digit stock code
        categories: list of category names to compute (None = all)
    Returns:
        dict: factor_name → value
    """
    all_factors = {}

    # News sentiment
    if categories is None or "news_sentiment" in categories:
        try:
            nf = NewsSentimentFactors()
            all_factors.update(nf.compute_all(code))
        except Exception:
            pass

    # Search & social heat
    if categories is None or "social_heat" in categories:
        try:
            sf = SearchHeatFactors()
            all_factors.update(sf.compute_all(code))
        except Exception:
            pass

    # Fund flow
    if categories is None or "fund_flow" in categories:
        try:
            ff = FundFlowFactors()
            all_factors.update(ff.compute_all(code))
        except Exception:
            pass

    # Auto-discovery: compute all registered factors
    try:
        extra = factor_registry.compute_all_factors(code)
        # Don't overwrite already-computed values
        for k, v in extra.items():
            if k not in all_factors:
                all_factors[k] = v
    except Exception:
        pass

    return all_factors


def compute_event_factors_batch(codes, max_stocks=None, categories=None):
    """
    Batch compute event factors for multiple stocks.

    Args:
        codes: list of stock codes
        max_stocks: limit to N stocks (for speed)
        categories: filter factor categories

    Returns:
        pd.DataFrame: index=code, columns=factor_names
    """
    if max_stocks:
        codes = codes[:max_stocks]

    print(f"Event Factor Engine V1.0 — Computing for {len(codes)} stocks...")
    print(f"  Registered factors: {len(factor_registry.list_factors())}")

    rows = []
    for i, code in enumerate(codes):
        try:
            factors = compute_event_factors_for_stock(code, categories)
            factors["code"] = str(code).zfill(6)
            rows.append(factors)
        except Exception as e:
            pass

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(codes)} stocks done...")

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("code")

    n_factors = len(result.columns) if not result.empty else 0
    print(f"  Complete: {len(result)} stocks, {n_factors} event factors")
    return result


# ============================================================
# Auto-Discovery: IC Test for New Factors
# ============================================================

def auto_test_factor(factor_name, codes, feature_dir="features",
                     min_samples=30):
    """
    Automatically test a factor's predictive power.

    1. Compute factor values for all stocks
    2. Align with future_return from feature files
    3. Calculate Spearman IC
    4. Report significance

    Returns:
        dict with ic_mean, ic_std, hit_rate, is_significant
    """
    from scipy.stats import spearmanr

    if factor_registry.get_factor(factor_name) is None:
        return {"error": f"Factor '{factor_name}' not registered"}

    print(f"Auto-testing factor: {factor_name}")

    # Compute factor values
    factor_values = {}
    for code in codes:
        val = factor_registry.compute_factor(factor_name, code)
        if not np.isnan(val):
            factor_values[code] = float(val)

    if len(factor_values) < min_samples:
        return {"error": f"Only {len(factor_values)} valid samples (need {min_samples})"}

    # Align with future returns
    aligned_factors = []
    aligned_returns = []

    for code, fval in factor_values.items():
        feat_file = os.path.join(feature_dir, f"{code}.csv")
        if not os.path.exists(feat_file):
            continue
        try:
            df = pd.read_csv(feat_file, encoding="utf-8-sig")
            if "future_return" not in df.columns or len(df) < 60:
                continue
            last_return = df["future_return"].iloc[-1]
            if not np.isnan(last_return):
                aligned_factors.append(fval)
                aligned_returns.append(last_return)
        except Exception:
            continue

    if len(aligned_factors) < min_samples:
        return {"error": f"Only {len(aligned_factors)} aligned samples"}

    # IC computation
    ic, pval = spearmanr(
        pd.Series(aligned_factors).rank(),
        pd.Series(aligned_returns).rank()
    )

    return {
        "factor": factor_name,
        "ic": round(float(ic), 5),
        "p_value": round(float(pval), 5),
        "abs_ic": round(abs(float(ic)), 5),
        "n_samples": len(aligned_factors),
        "is_significant": abs(ic) > 0.02,
        "direction": "positive" if ic > 0 else "negative",
    }


def auto_discover_new_factors(codes, test_all_registered=True,
                              custom_factors=None):
    """
    Automatically test all registered (or custom) factors.
    Returns ranked list of factor test results.
    """
    if test_all_registered:
        factor_names = [f["name"] for f in factor_registry.list_factors()]
    else:
        factor_names = custom_factors or []

    results = []
    for fname in factor_names:
        result = auto_test_factor(fname, codes)
        if "error" not in result:
            results.append(result)
        else:
            print(f"  {fname}: {result['error']}")

    # Sort by |IC|
    results.sort(key=lambda r: r["abs_ic"], reverse=True)

    print(f"\nAuto-Discovery Results ({len(results)} factors tested):")
    print(f"  {'Factor':<25s} {'|IC|':>7s} {'IC':>8s} {'p-val':>8s} {'N':>6s} {'Sig?'}")
    for r in results:
        ic_str = f"{r['ic']:+8.4f}" if r['ic'] != 0 else "  0.0000"
        sig = "★" if r["is_significant"] else ""
        print(f"  {r['factor']:<25s} {r['abs_ic']:7.4f} {ic_str} "
              f"{r['p_value']:8.4f} {r['n_samples']:6d} {sig}")

    return results


# ============================================================
# User-Friendly: Register Custom Factor
# ============================================================

def register_custom_factor(name, category, description, compute_fn):
    """
    Public API to register a custom alternative factor.

    Example:
        >>> def my_factor(code, **params):
        ...     # Your logic here — fetch any data, compute any signal
        ...     return some_float_value
        >>> register_custom_factor("my_custom", "custom", "My custom factor", my_factor)
    """
    factor_registry.register(name, category, description, compute_fn)
    print(f"Registered custom factor: {name} [{category}]")


def list_registered_factors():
    """List all factors in the auto-discovery registry."""
    factors = factor_registry.list_factors()
    for f in factors:
        print(f"  {f['name']:<30s} [{f['category']:<15s}] {f['description']}")
    print(f"\n  Total: {len(factors)} factors")
    return factors


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Demo: list all registered factors
    print("=" * 60)
    print("Event & Alternative Factor Engine V1.0")
    print("=" * 60)
    list_registered_factors()

    # Demo: test on a few stocks
    print("\n" + "=" * 60)
    print("Testing on sample stocks...")
    print("=" * 60)

    # Get stock list
    stock_list_path = "data/stock_list.csv"
    if os.path.exists(stock_list_path):
        stocks = pd.read_csv(stock_list_path, dtype={"code": str})
        codes = stocks["code"].str.zfill(6).tolist()[:20]  # First 20 for demo
    else:
        codes = ["000001", "600519", "000858", "002594"][:10]

    # Compute event factors
    df = compute_event_factors_batch(codes)
    if not df.empty:
        print(f"\nSample output ({len(df)} stocks):")
        print(df.head().to_string())

    # Auto-discovery IC test
    print("\n" + "=" * 60)
    print("Auto-Discovery: IC Testing")
    print("=" * 60)
    auto_discover_new_factors(codes)
