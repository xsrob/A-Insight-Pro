"""
A-Insight Pro
股票价格数据管理 V5
"""

import os
import pandas as pd


DATA_DIR = "data"


# =========================
# 读取单股票价格
# =========================

def load_price(code):

    file = os.path.join(
        DATA_DIR,
        f"{code}.csv"
    )

    if not os.path.exists(file):

        print(
            "不存在:",
            file
        )

        return None


    df = pd.read_csv(
        file
    )


    df["date"] = pd.to_datetime(
        df["date"]
    )


    df = df.sort_values(
        "date"
    )


    return df



# =========================
# 获取最新价格
# =========================

def latest_price(code):

    df = load_price(code)


    if df is None:

        return None


    row = df.iloc[-1]


    return {

        "code": code,

        "date":
        str(row["date"].date()),

        "close":
        float(row["close"]),

        "volume":
        float(row["volume"])

    }



# =========================
# 股票列表
# =========================

def get_all_stock():

    files = []


    for f in os.listdir(DATA_DIR):

        if f.endswith(".csv"):

            files.append(
                f.replace(
                    ".csv",
                    ""
                )
            )


    return sorted(files)



# =========================
# 数据统计
# =========================

def summary():


    stocks = get_all_stock()


    print(
        "股票数量:",
        len(stocks)
    )


    total = 0


    for code in stocks:

        df = load_price(code)


        if df is not None:

            total += len(df)



    print(
        "总数据量:",
        total
    )




if __name__=="__main__":


    summary()