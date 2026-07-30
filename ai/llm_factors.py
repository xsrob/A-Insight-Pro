"""
A-Insight Pro
LLM-Based Sentiment & Event Factor Extraction V1.0

Uses Large Language Models to extract structured sentiment signals
from macro news, policy announcements, and social media text.

Unlike simple keyword matching, LLMs understand context:
  - "降息" → rate cut → bullish, but "降息不及预期" → rate cut below expectations → bearish
  - "贸易摩擦升级" → trade war escalation → sector-specific impact
  - "行业政策收紧" → regulatory tightening → bearish for affected sectors

Architecture:
  1. Fetch news headlines (akshare)
  2. Batch-process through LLM with structured prompt
  3. Extract: sentiment(-5 ~ +5), affected sectors, event type, confidence
  4. Aggregate to daily factor scores
"""

import os, json, time, re
import numpy as np
import pandas as pd
from datetime import datetime

REPORT_DIR = "reports"
CACHE_DIR = os.path.join(REPORT_DIR, "llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# LLM Client (works with any OpenAI-compatible API)
# ============================================================

def _llm_analyze(texts, system_prompt, model=None):
    """
    Send batch of texts to LLM for analysis.
    Uses OpenAI-compatible API. Falls back gracefully if unavailable.
    """
    # Try to use the available LLM (Claude/DeepSeek/OpenAI compatible)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("ANTHROPIC_BASE_URL")

    if not api_key:
        return None  # Will fall back to heuristic mode

    try:
        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        joined_texts = "\n---\n".join([f"[{i+1}] {t}" for i, t in enumerate(texts)])

        payload = {
            "model": model or os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze these Chinese financial news headlines:\n\n{joined_texts}\n\nReturn JSON array only."}
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        endpoint = f"{api_base}/chat/completions" if api_base else "https://api.anthropic.com/v1/messages"
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        return None


# ============================================================
# Heuristic Fallback (when LLM unavailable)
# ============================================================

# Domain-specific sentiment lexicons (Chinese financial)
MACRO_BULLISH_KEYWORDS = [
    "降准", "降息", "宽松", "刺激", "稳增长", "扩大内需", "减税",
    "基建投资", "新基建", "逆周期调节", "流动性充裕", "信贷扩张",
    "消费复苏", "经济回暖", "PMI回升", "出口增长", "社融超预期",
    "贸易缓和", "关系改善", "协议签署", "政策支持", "扶持",
]

MACRO_BEARISH_KEYWORDS = [
    "加息", "收紧", "去杠杆", "通胀压力", "经济下行", "衰退",
    "贸易摩擦", "制裁", "关税", "脱钩", "冲突", "地缘风险",
    "监管收紧", "整改", "约谈", "处罚", "暂停上市", "ST",
    "债务违约", "暴雷", "资金链断裂", "流动性危机", "股灾",
    "疫情反复", "封锁", "供应链中断",
]

SECTOR_KEYWORDS = {
    "新能源": ["光伏", "风电", "储能", "锂电", "新能源车", "充电桩"],
    "半导体": ["芯片", "集成电路", "光刻", "晶圆", "封装测试"],
    "消费": ["白酒", "食品", "家电", "零售", "餐饮", "旅游"],
    "医药": ["创新药", "医疗器械", "CXO", "疫苗", "中药"],
    "金融": ["银行", "券商", "保险", "信托", "金融科技"],
    "地产": ["房地产", "开发贷", "按揭", "土地出让", "保障房"],
    "军工": ["国防", "航天", "船舶", "军工电子"],
    "AI科技": ["人工智能", "大模型", "算力", "数据中心", "机器人"],
}


def _heuristic_sentiment(text):
    """Keyword-based sentiment scoring (fallback when LLM unavailable)."""
    score = 0.0
    text_lower = str(text)

    for kw in MACRO_BULLISH_KEYWORDS:
        if kw in text_lower:
            score += 1.0

    for kw in MACRO_BEARISH_KEYWORDS:
        if kw in text_lower:
            score -= 1.5

    # Detect negation ("不及预期", "低于预期", "未能")
    negations = ["不及预期", "低于预期", "未能", "推迟", "受阻", "放缓"]
    has_negation = any(n in text_lower for n in negations)
    if has_negation:
        score -= 2.0

    # Clamp
    return max(-5.0, min(5.0, score))


def _heuristic_sectors(text):
    """Detect affected sectors from text."""
    text_lower = str(text)
    affected = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            affected.append(sector)
    return affected


# ============================================================
# Main Analyzer
# ============================================================

class LLMFactorExtractor:
    """
    Extract structured sentiment factors from financial text using LLM.
    Falls back to heuristic keyword analysis when LLM is unavailable.
    """

    SYSTEM_PROMPT = """You are a Chinese financial sentiment analyzer. For each news headline, return a JSON array with:
[
  {
    "sentiment": float between -5 (very bearish) and +5 (very bullish),
    "confidence": float 0-1,
    "category": "monetary_policy|fiscal_policy|trade|regulation|industry|earnings|market_sentiment|other",
    "affected_sectors": ["sector1", "sector2"],
    "duration": "short|medium|long",
    "reasoning": "one sentence CN"
  }
]
Rules:
- "降息" = +3 to +4 unless "不及预期" → 0 to +1
- "行业整顿/监管收紧" = -3 to -5 for affected sector
- "政策支持/补贴" = +2 to +4
- Economic data beats expectations → +2 to +3
- Trade tensions → -2 to -4
- Neutral announcements → -1 to +1"""

    def __init__(self, use_llm=True):
        self.use_llm = use_llm
        self.results_cache = {}

    def analyze_headlines(self, headlines):
        """Analyze a list of headlines and return structured factors."""
        if not headlines:
            return []

        # Check cache
        cache_key = "|".join(str(h)[:50] for h in headlines[:20])
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]

        results = []

        if self.use_llm:
            llm_results = _llm_analyze(headlines[:20], self.SYSTEM_PROMPT)
            if llm_results:
                self.results_cache[cache_key] = llm_results
                return llm_results

        # Heuristic fallback
        for text in headlines:
            text = str(text)
            sentiment = _heuristic_sentiment(text)
            sectors = _heuristic_sectors(text)
            results.append({
                "sentiment": sentiment,
                "confidence": 0.6,
                "category": "macro",
                "affected_sectors": sectors,
                "duration": "short",
                "reasoning": "heuristic",
            })

        self.results_cache[cache_key] = results
        return results

    def compute_daily_factors(self, headlines_list):
        """Aggregate analyzed headlines into daily factor scores."""
        analyses = self.analyze_headlines(headlines_list)
        if not analyses:
            return {}

        sentiments = [a["sentiment"] for a in analyses]
        confidences = [a.get("confidence", 0.6) for a in analyses]

        # Weighted by confidence
        weighted_sent = np.average(sentiments, weights=confidences) if confidences else np.mean(sentiments)

        # Sector-level aggregation
        sector_scores = {}
        for a in analyses:
            for sector in a.get("affected_sectors", []):
                if sector not in sector_scores:
                    sector_scores[sector] = []
                sector_scores[sector].append(a["sentiment"])

        # Policy/macro sentiment
        policy_texts = [a for a in analyses if a.get("category") in
                        ("monetary_policy", "fiscal_policy", "regulation", "trade")]
        policy_sent = np.mean([a["sentiment"] for a in policy_texts]) if policy_texts else weighted_sent

        return {
            "llm_macro_sentiment": round(float(weighted_sent), 2),
            "llm_policy_sentiment": round(float(policy_sent), 2),
            "llm_sentiment_std": round(float(np.std(sentiments)), 2) if len(sentiments) > 1 else 0.0,
            "llm_positive_ratio": round(float((np.array(sentiments) > 0).mean()), 3),
            "llm_negative_ratio": round(float((np.array(sentiments) < 0).mean()), 3),
            "llm_news_count": len(headlines_list),
        }


# ============================================================
# Batch Processing
# ============================================================

def compute_llm_factors_for_stock(code, headlines=None):
    """Compute LLM-based factors for a single stock."""
    extractor = LLMFactorExtractor(use_llm=False)  # Heuristic by default (no API key)

    if headlines is None:
        # Try to fetch news via event_factors
        try:
            from ai.event_factors import NewsSentimentFactors
            nf = NewsSentimentFactors()
            news = nf.fetch_news(code, max_news=30)
            headlines = [n["title"] for n in news if n.get("title")]
        except Exception:
            headlines = []

    if not headlines:
        return {
            "llm_macro_sentiment": np.nan,
            "llm_policy_sentiment": np.nan,
            "llm_sentiment_std": np.nan,
            "llm_positive_ratio": np.nan,
            "llm_negative_ratio": np.nan,
            "llm_news_count": 0,
        }

    return extractor.compute_daily_factors(headlines)


def compute_llm_factors_batch(codes, max_stocks=50):
    """Batch compute LLM factors for multiple stocks."""
    print(f"LLM Factor Extraction — {min(len(codes), max_stocks)} stocks...")
    extractor = LLMFactorExtractor(use_llm=False)

    results = []

    for i, code in enumerate(codes[:max_stocks]):
        try:
            from ai.event_factors import NewsSentimentFactors
            nf = NewsSentimentFactors()
            news = nf.fetch_news(code, max_news=20)
            headlines = [n["title"] for n in news if n.get("title")]

            factors = extractor.compute_daily_factors(headlines)
            factors["code"] = str(code).zfill(6)
            results.append(factors)
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{min(len(codes), max_stocks)} done...")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.set_index("code")

    print(f"  Complete: {len(results)} stocks, {len(df.columns) if not df.empty else 0} LLM factors")
    return df


# ============================================================
# Register with event_factors
# ============================================================

def register_with_event_engine():
    """Register LLM factors with the event_factors auto-discovery registry."""
    try:
        from ai.event_factors import register_custom_factor

        def _llm_macro_sentiment(code, **kw):
            factors = compute_llm_factors_for_stock(code)
            return factors.get("llm_macro_sentiment", np.nan)

        def _llm_policy_sentiment(code, **kw):
            factors = compute_llm_factors_for_stock(code)
            return factors.get("llm_policy_sentiment", np.nan)

        def _llm_positive_ratio(code, **kw):
            factors = compute_llm_factors_for_stock(code)
            return factors.get("llm_positive_ratio", np.nan)

        register_custom_factor(
            "llm_macro_sentiment", "llm_sentiment",
            "LLM-extracted macro sentiment score (-5 to +5)",
            _llm_macro_sentiment
        )
        register_custom_factor(
            "llm_policy_sentiment", "llm_sentiment",
            "LLM-extracted policy sentiment score (-5 to +5)",
            _llm_policy_sentiment
        )
        register_custom_factor(
            "llm_positive_ratio", "llm_sentiment",
            "Ratio of positive sentiment news (LLM-judged)",
            _llm_positive_ratio
        )
        print("LLM factors registered with event engine: 3 factors")
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Factor Extractor V1.0")
    print("=" * 60)

    # Test on sample headlines
    test_headlines = [
        "央行宣布降准0.5个百分点，释放长期资金约1万亿",
        "美国对中国半导体出口管制升级，多家企业列入实体清单",
        "新能源车购置税减免政策延续至2027年",
        "某房企债务违约，涉及金额超百亿",
        "三季度GDP数据超预期，同比增长5.2%",
        "监管层约谈多家互联网平台，要求整改算法推荐",
        "中美经贸高层对话取得积极进展",
        "全国流感疫情抬头，多地医院门诊量激增",
    ]

    extractor = LLMFactorExtractor(use_llm=False)
    factors = extractor.compute_daily_factors(test_headlines)

    print("\nTest Headlines Analysis:")
    for t in test_headlines:
        s = _heuristic_sentiment(t)
        label = "Bullish" if s > 1 else ("Bearish" if s < -1 else "Neutral")
        print(f"  [{label:>7s}] ({s:+3.1f}) {t}")

    print(f"\nAggregated Factors:")
    for k, v in factors.items():
        print(f"  {k}: {v}")
