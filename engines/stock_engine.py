"""
A-Insight Pro
股票分析引擎 V3
"""

import pandas as pd


class StockEngine:


    def __init__(self):
        pass



    def analyze(self, df):

        data = df.copy()

        score = 50

        signals = []

        risks = []


        # =========================
        # 数据检查
        # =========================

        if len(data) < 60:

            return {
                "score": 40,
                "signal": "数据不足",
                "signals": ["历史数据不足"],
                "risks": []
            }



        close = data["close"]

        volume = data["volume"]



        # =========================
        # 均线系统
        # =========================

        data["MA5"] = close.rolling(5).mean()

        data["MA20"] = close.rolling(20).mean()

        data["MA60"] = close.rolling(60).mean()



        latest = data.iloc[-1]



        # 短期趋势

        if latest["MA5"] > latest["MA20"]:

            score += 6

            signals.append(
                "短期均线向上"
            )

        else:

            score -= 4

            risks.append(
                "短期趋势偏弱"
            )



        # 中期趋势

        if latest["MA20"] > latest["MA60"]:

            score += 8

            signals.append(
                "中期趋势向上"
            )

        else:

            score -= 6

            risks.append(
                "中期趋势不足"
            )



        # 股价位置

        if latest["close"] > latest["MA20"]:

            score += 6

            signals.append(
                "价格站上20日均线"
            )

        else:

            score -= 5



        # =========================
        # 动能分析
        # =========================


        change_5 = (
            latest["close"]
            /
            data.iloc[-6]["close"]
            -
            1
        )


        change_20 = (
            latest["close"]
            /
            data.iloc[-21]["close"]
            -
            1
        )



        if change_5 > 0:

            score += 5

            signals.append(
                "近5日上涨"
            )

        else:

            score -= 3



        if change_20 > 0:

            score += 8

            signals.append(
                "20日趋势上涨"
            )

        else:

            score -= 5



        # 涨幅过热

        if change_20 > 0.25:

            score -= 8

            risks.append(
                "短期涨幅过高"
            )



        # =========================
        # 成交量资金
        # =========================


        volume_avg = (
            volume
            .rolling(20)
            .mean()
            .iloc[-1]
        )


        if latest["volume"] > volume_avg:

            score += 8

            signals.append(
                "成交量活跃"
            )

        else:

            score -= 2

            risks.append(
                "成交量不足"
            )



        # 量价配合

        if change_5 > 0 and latest["volume"] > volume_avg:

            score += 5

            signals.append(
                "量价同步"
            )



        # =========================
        # 波动风险
        # =========================


        volatility = (
            close
            .pct_change()
            .rolling(20)
            .std()
            .iloc[-1]
        )



        if volatility < 0.03:

            score += 5

            signals.append(
                "波动稳定"
            )

        elif volatility > 0.06:

            score -= 8

            risks.append(
                "波动较大"
            )



        # =========================
        # 回撤控制
        # =========================


        high_20 = (
            close
            .rolling(20)
            .max()
            .iloc[-1]
        )


        drawdown = (
            latest["close"]
            /
            high_20
            -
            1
        )



        if drawdown > -0.08:

            score += 3

        else:

            score -= 5

            risks.append(
                "近期回撤明显"
            )



        # =========================
        # 突破判断
        # =========================


        high_previous = (
            close
            .rolling(20)
            .max()
            .shift(1)
            .iloc[-1]
        )


        if latest["close"] > high_previous:

            score += 8

            signals.append(
                "突破近期高点"
            )



        # =========================
        # RSI
        # =========================


        delta = close.diff()


        gain = (
            delta
            .where(delta > 0, 0)
            .rolling(14)
            .mean()
        )


        loss = (
            -delta
            .where(delta < 0, 0)
            .rolling(14)
            .mean()
        )


        rs = gain / loss


        rsi = 100 - (100 / (1 + rs))


        latest_rsi = rsi.iloc[-1]



        if latest_rsi > 80:

            score -= 8

            risks.append(
                "RSI过热"
            )


        elif latest_rsi < 30:

            score += 3

            signals.append(
                "超跌反弹可能"
            )



        # =========================
        # 分数限制
        # =========================

        score = int(
            max(
                0,
                min(
                    100,
                    score
                )
            )
        )



        # =========================
        # 评级
        # =========================


        if score >= 85:

            signal = "强烈关注"


        elif score >= 75:

            signal = "关注"


        elif score >= 60:

            signal = "观察"


        else:

            signal = "回避"



        return {

            "score": score,

            "signal": signal,

            "signals": signals,

            "risks": risks

        }