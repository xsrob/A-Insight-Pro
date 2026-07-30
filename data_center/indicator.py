"""
A-Insight Pro
技术指标计算 V1
"""


import os
import pandas as pd
import numpy as np


DATA_DIR="data"



def calculate_indicator(df):


    df=df.copy()


    # 涨跌幅

    df["change_pct"] = (
        df["close"]
        .pct_change()
        *100
    )



    # 均线

    for n in [5,10,20,60]:

        df[f"ma{n}"] = (

            df["close"]

            .rolling(n)

            .mean()

        )



    # EMA

    ema12=df["close"].ewm(
        span=12,
        adjust=False
    ).mean()


    ema26=df["close"].ewm(
        span=26,
        adjust=False
    ).mean()



    df["ema12"]=ema12

    df["ema26"]=ema26



    # MACD


    df["macd"]=ema12-ema26


    df["macd_signal"]=(
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )


    df["macd_hist"]=(
        df["macd"]
        -
        df["macd_signal"]
    )



    # RSI


    delta=df["close"].diff()


    gain=(
        delta
        .where(delta>0,0)
    )


    loss=(
        -delta
        .where(delta<0,0)
    )


    avg_gain=(
        gain
        .rolling(14)
        .mean()
    )


    avg_loss=(
        loss
        .rolling(14)
        .mean()
    )


    rs=avg_gain/avg_loss


    df["rsi"]=100-(100/(1+rs))



    # BOLL


    mid=(
        df["close"]
        .rolling(20)
        .mean()
    )


    std=(
        df["close"]
        .rolling(20)
        .std()
    )


    df["boll_mid"]=mid

    df["boll_upper"]=mid+2*std

    df["boll_lower"]=mid-2*std



    # 成交量变化


    df["vol_change"]=(
        df["volume"]
        .pct_change()
        *100
    )



    return df





def process_all():


    files=[

        f for f in os.listdir(DATA_DIR)

        if f.endswith(".csv")

    ]



    success=0



    for f in files:


        path=os.path.join(
            DATA_DIR,
            f
        )


        print(
            "处理:",
            f
        )


        df=pd.read_csv(path)



        df=calculate_indicator(df)



        df.to_csv(

            path,

            index=False,

            encoding="utf-8-sig"

        )


        success+=1



    print("================")

    print(

        "指标计算完成:",

        success

    )





if __name__=="__main__":

    process_all()