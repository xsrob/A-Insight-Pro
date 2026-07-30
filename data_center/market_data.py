"""
A-Insight Pro
市场数据中心
"""

from data_center.index_data import get_shanghai_index
from data_center.stock_data import get_stock_list



def get_market_snapshot():

    """
    获取市场快照
    """

    print("正在生成市场快照...")


    # 指数数据

    index_data = get_shanghai_index()


    # 股票列表

    stocks = get_stock_list()


    snapshot = {

        "index":

            index_data.tail(1),


        "stock_count":

            len(stocks) if stocks is not None else 0

    }


    return snapshot



if __name__ == "__main__":


    market = get_market_snapshot()


    print("====================")

    print("市场股票数量:")

    print(
        market["stock_count"]
    )


    print("====================")

    print("最新指数:")

    print(
        market["index"]
    )