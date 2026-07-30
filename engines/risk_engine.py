"""
A-Insight Pro
风险评估引擎
"""


class RiskEngine:


    def analyze(self, market_result):


        score = market_result["market_score"]


        result = {}


        if score >= 80:

            risk = "低风险"

            position = "80%-100%"


        elif score >= 65:

            risk = "中低风险"

            position = "60%-80%"


        elif score >= 45:

            risk = "中等风险"

            position = "40%-60%"


        else:

            risk = "高风险"

            position = "20%-30%"



        result["risk_level"] = risk

        result["suggested_position"] = position


        return result