"""
A-Insight Pro
股票排名引擎
"""

import pandas as pd

from sqlalchemy.orm import Session

from database.database import get_engine
from database.models import StockList, StockPrice

from engines.stock_engine import StockEngine


class RankingEngine:


    def __init__(self):

        self.stock_engine = StockEngine()



    def run(self, limit=20):

        """
        扫描股票并排名

        limit:
            输出前多少名
        """


        engine = get_engine()

        session = Session(engine)


        results = []


        print(
            "开始扫描股票..."
        )



        # =========================
        # 获取股票列表
        # 扫描500只
        # =========================

        stocks = session.query(
            StockList
        ).limit(500).all()



        total = len(stocks)


        print(
            "扫描数量:",
            total
        )



        for index, stock in enumerate(stocks):


            print(
                f"{index + 1}/{total}",
                stock.code,
                stock.name
            )



            prices = session.query(
                StockPrice
            ).filter(
                StockPrice.code == stock.code
            ).order_by(
                StockPrice.date
            ).all()



            # 少于20天数据跳过

            if len(prices) < 20:

                continue



            df = pd.DataFrame(
                [
                    {
                        "date": x.date,
                        "open": x.open,
                        "high": x.high,
                        "low": x.low,
                        "close": x.close,
                        "volume": x.volume
                    }

                    for x in prices
                ]
            )



            try:

                result = self.stock_engine.analyze(
                    df
                )


                results.append(
                    {
                        "code": stock.code,

                        "name": stock.name,

                        "score": result["score"],

                        "signal": result["signal"]
                    }
                )


            except Exception as e:

                print(
                    stock.code,
                    "分析失败:",
                    e
                )

                continue



        session.close()



        # =========================
        # 排序
        # =========================

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )



        return results[:limit]





if __name__ == "__main__":


    engine = RankingEngine()



    ranking = engine.run(
        limit=20
    )



    print(
        "\n===================="
    )



    print(
        "股票排名结果:"
    )



    for i, item in enumerate(
        ranking,
        start=1
    ):

        print(
            i,
            item
        )