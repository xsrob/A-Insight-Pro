"""
A-Insight Pro
指数数据模块
"""

import akshare as ak


def get_shanghai_index():

    """
    获取上证指数历史数据
    """

    data = ak.stock_zh_index_daily(
        symbol="sh000001"
    )

    return data


if __name__ == "__main__":

    print("正在获取上证指数数据...")

    index_data = get_shanghai_index()

    print(index_data.tail())