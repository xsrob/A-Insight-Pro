"""
A-Insight Pro
股票评分引擎
"""


class ScoreEngine:


    def calculate(self, stock_data):


        result = []


        for index, row in stock_data.iterrows():


            score = 50


            # 后续加入:
            # 趋势
            # 成交量
            # 技术指标
            # 资金流


            result.append({

                "code": row["code"],

                "name": row["name"],

                "score": score

            })


        return result