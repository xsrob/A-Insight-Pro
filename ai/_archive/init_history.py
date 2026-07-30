"""
A-Insight Pro
历史行情初始化系统 V1.0

功能:
- 首次恢复股票历史K线
- 生成features
"""

import os
import time
import random
import requests
import pandas as pd


DATA_DIR = "data"
FEATURE_DIR = "features"

STOCK_LIST = "data/stock_list.csv"


def init():

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)



def get_history(code):

    """
    新浪历史日K接口
    """

    code = str(code).zfill(6)


    if code.startswith(("6","5")):

        symbol = "sh" + code

    else:

        symbol = "sz" + code


    url = (
        "https://quotes.sina.cn/cn/api/jsonp_v2.php/"
        "var%20_data=/CN_MarketDataService.getKLineData"
        "?symbol="
        + symbol +
        "&scale=240"
        "&ma=no"
        "&datalen=1000"
    )


    try:

        r = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )


        text=r.text


        start=text.find("[")

        end=text.rfind("]")


        if start<0 or end<0:

            return None


        data=text[start:end+1]


        rows=pd.read_json(data)


        return rows


    except Exception:

        return None




def make_features(df):


    df=df.copy()


    df["return"] = (
        df["close"]
        .pct_change()
    )


    df["ma5"] = (
        df["close"]
        .rolling(5)
        .mean()
    )


    df["ma10"] = (
        df["close"]
        .rolling(10)
        .mean()
    )


    df["ma20"] = (
        df["close"]
        .rolling(20)
        .mean()
    )


    df["ma60"] = (
        df["close"]
        .rolling(60)
        .mean()
    )


    df["volume_change"] = (
        df["volume"]
        .pct_change()
    )


    df["volatility"] = (
        df["return"]
        .rolling(20)
        .std()
    )


    delta=df["close"].diff()


    gain=(
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )


    loss=(
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )


    rs=gain/loss.replace(0,1e-9)


    df["rsi"]=100-(100/(1+rs))


    ema12=df["close"].ewm(
        span=12,
        adjust=False
    ).mean()


    ema26=df["close"].ewm(
        span=26,
        adjust=False
    ).mean()


    df["macd"]=ema12-ema26


    return df.dropna()




def save(code,df):


    code=str(code).zfill(6)


    df.to_csv(
        f"{DATA_DIR}/{code}.csv",
        index=False,
        encoding="utf-8-sig"
    )


    feature=make_features(df)


    feature.to_csv(
        f"{FEATURE_DIR}/{code}.csv",
        index=False,
        encoding="utf-8-sig"
    )




def run():


    init()


    stocks=pd.read_csv(
        STOCK_LIST,
        dtype={
            "code":str
        }
    )


    stocks["code"]=(
        stocks["code"]
        .str.zfill(6)
    )


    total=len(stocks)


    print("================")
    print("历史行情初始化")
    print("================")


    for i,row in stocks.iterrows():


        code=row["code"]


        print(
            f"[{i+1}/{total}] {code}"
        )


        df=get_history(code)


        if df is None:

            print(
                code,
                "失败"
            )

            continue



        try:


            df=df.rename(
                columns={
                    "day":"date",
                    "open":"open",
                    "high":"high",
                    "low":"low",
                    "close":"close",
                    "volume":"volume"
                }
            )


            df=df[
                [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]
            ]


            save(
                code,
                df
            )


            print(
                code,
                "完成",
                len(df),
                "条"
            )


        except Exception as e:

            print(
                code,
                e
            )


        time.sleep(
            random.uniform(
                0.5,
                1
            )
        )


    print("================")
    print("历史初始化完成")
    print("================")



if __name__=="__main__":

    run()