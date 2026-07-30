"""
A-Insight Pro
A股股票池生成模块 V1
"""

import os
import pandas as pd
import akshare as ak


DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)


def create_stock_list():

    print("正在获取A股股票列表...")


    try:

        df = ak.stock_info_a_code_name()


        df["code"] = (
            df["code"]
            .astype(str)
            .str.zfill(6)
        )


        # 保存前500只
        df = df.head(500)


        file = os.path.join(
            DATA_DIR,
            "stock_list.csv"
        )


        df.to_csv(
            file,
            index=False,
            encoding="utf-8-sig"
        )


        print("================")
        print(
            "股票池生成完成:",
            len(df)
        )

        print(
            "保存:",
            file
        )


        return df


    except Exception as e:

        print(
            "生成失败:",
            e
        )

        return None



if __name__ == "__main__":

    create_stock_list()