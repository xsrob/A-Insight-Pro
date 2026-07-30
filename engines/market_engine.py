"""
A-Insight Pro
市场趋势分析引擎 v2
"""


class MarketEngine:


    def analyze(self, index_data):

        data = index_data.copy()

        score = 50

        signals = []


        # 均线计算

        data["MA5"] = (
            data["close"]
            .rolling(5)
            .mean()
        )


        data["MA20"] = (
            data["close"]
            .rolling(20)
            .mean()
        )


        latest = data.iloc[-1]


        # 趋势判断

        if latest["MA5"] > latest["MA20"]:

            score += 20

            signals.append(
                "短期均线高于长期均线，趋势向上"
            )

        else:

            score -= 20

            signals.append(
                "短期均线低于长期均线，趋势偏弱"
            )


        # 最近走势

        change = (
            latest["close"]
            -
            data.iloc[-5]["close"]
        )


        if change > 0:

            score += 10

            signals.append(
                "近5日指数上涨，市场动能增强"
            )

        else:

            score -= 10

            signals.append(
                "近5日指数下跌，市场动能减弱"
            )


        # 成交量

        volume_now = latest["volume"]

        volume_avg = (
            data["volume"]
            .tail(20)
            .mean()
        )


        if volume_now > volume_avg:

            score += 10

            signals.append(
                "成交量高于近期平均，资金活跃"
            )

        else:

            score -= 5

            signals.append(
                "成交量低于近期平均，资金谨慎"
            )


        score = max(
            0,
            min(
                100,
                score
            )
        )


        if score >= 80:

            trend = "强势上涨"

        elif score >= 65:

            trend = "偏强"

        elif score >= 45:

            trend = "震荡"

        else:

            trend = "偏弱"



        return {

            "trend": trend,

            "market_score": score,

            "signals": signals

        }