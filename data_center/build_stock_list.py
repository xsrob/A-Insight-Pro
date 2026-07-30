"""
A-Insight Pro
股票池自动生成
"""

import os
import pandas as pd


DATA_DIR = "data"


def build_stock_list():

    files = os.listdir(DATA_DIR)

    stocks = []


    for f in files:

        if not f.endswith(".csv"):
            continue

        if f == "stock_list.csv":
            continue


        code = f.replace(".csv","")


        # 只保留6位股票代码
        if len(code) != 6:
            continue


        path = os.path.join(
            DATA_DIR,
            f
        )


        try:

            df = pd.read_csv(
                path,
                nrows=1
            )


            stocks.append(
                {
                    "code":code,
                    "name":""
                }
            )


            print(
                code,
                "OK"
            )


        except Exception as e:

            print(
                code,
                "失败",
                e
            )



    result = pd.DataFrame(
        stocks
    )


    result = result.sort_values(
        "code"
    )


    output = os.path.join(
        DATA_DIR,
        "stock_list.csv"
    )


    result.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )


    print("================")
    print(
        "股票数量:",
        len(result)
    )

    print(
        "生成:",
        output
    )



if __name__=="__main__":

    build_stock_list()