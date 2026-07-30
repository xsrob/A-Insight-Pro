"""
A-Insight Pro
数据清洗 V1
"""

import os
import pandas as pd
from datetime import datetime


DATA_DIR="data"


def clean_file(path):

    print("清洗:", path)


    df=pd.read_csv(path)


    # 日期转换

    df["date"]=pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    # 删除无效日期

    df=df.dropna(
        subset=["date"]
    )


    # 删除未来数据

    today=pd.Timestamp.today()


    df=df[
        df["date"]<=today
    ]


    # 去重

    df=df.drop_duplicates(
        subset=["date"]
    )


    # 排序

    df=df.sort_values(
        "date"
    )


    # 删除缺失

    df=df.dropna()


    # 价格异常

    df=df[
        (df["close"]>0)
        &
        (df["open"]>0)
    ]


    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "剩余:",
        len(df),
        "条"
    )



def main():


    files=[

        f for f in os.listdir(DATA_DIR)

        if f.endswith(".csv")

    ]


    count=0


    for f in files:

        clean_file(
            os.path.join(
                DATA_DIR,
                f
            )
        )

        count+=1


    print("================")
    print(
        "完成:",
        count
    )



if __name__=="__main__":

    main()