# A-Insight Pro
# update_data.py 自动生成器 V2.3.1

import os


content = r'''
"""
A-Insight Pro
行情自动更新系统 V2.3.1

数据源:
腾讯财经

功能:
- 自动更新股票价格
- 断点续跑
- 错误记录
- 保存data
- 生成features
"""


import os
import time
import random
import traceback
import requests
import pandas as pd


DATA_DIR = "data"
FEATURE_DIR = "features"
REPORT_DIR = "reports"

STOCK_LIST = "data/stock_list.csv"

PROGRESS_FILE = "reports/update_progress.txt"
ERROR_FILE = "reports/update_error.log"



def init():

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)



def log_error(msg):

    with open(
        ERROR_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            msg + "\\n"
        )



def market_code(code):

    code = str(code).zfill(6)

    if code.startswith(("5","6")):

        return "sh" + code

    return "sz" + code



def get_price(code):

    url = (
        "https://qt.gtimg.cn/q="
        +
        market_code(code)
    )


    headers = {

        "User-Agent":
        "Mozilla/5.0"

    }


    for i in range(3):

        try:

            r = requests.get(
                url,
                headers=headers,
                timeout=10
            )


            if r.text.startswith("v_"):

                return r.text


        except Exception as e:

            print(
                code,
                "失败",
                i+1
            )

            time.sleep(
                5
            )


    return None



def parse_price(text):

    try:

        data = (
            text
            .split('"')[1]
            .split("~")
        )


        return float(data[3])


    except:

        return None




def load_stock(code):

    file = os.path.join(
        DATA_DIR,
        str(code).zfill(6)+".csv"
    )


    if os.path.exists(file):

        try:

            return pd.read_csv(file)

        except:

            pass



    return pd.DataFrame(
        columns=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )




def update_stock(df, price):


    today = pd.Timestamp.now().strftime(
        "%Y-%m-%d"
    )


    row = {

        "date":today,
        "open":price,
        "high":price,
        "low":price,
        "close":price,
        "volume":0

    }



    if len(df)>0 and str(df.iloc[-1]["date"]) == today:

        df.loc[
            df.index[-1],
            "close"
        ] = price


    else:

        df = pd.concat(
            [
                df,
                pd.DataFrame([row])
            ],
            ignore_index=True
        )


    return df




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


    df["ma20"] = (
        df["close"]
        .rolling(20)
        .mean()
    )


    df["volatility"] = (
        df["return"]
        .rolling(20)
        .std()
    )


    return df.dropna()




def save_stock(code,df):


    code=str(code).zfill(6)


    df.to_csv(
        os.path.join(
            DATA_DIR,
            code+".csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )


    feature=make_features(df)


    feature.to_csv(
        os.path.join(
            FEATURE_DIR,
            code+".csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )




def get_progress():


    if os.path.exists(PROGRESS_FILE):

        return open(
            PROGRESS_FILE,
            encoding="utf-8"
        ).read().strip()


    return ""




def save_progress(code):

    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            str(code)
        )




def update():


    init()


    print("================")
    print(
        "A-Insight 行情更新系统 V2.3.1"
    )
    print(
        "数据源: 腾讯财经"
    )
    print("================")



    stocks=pd.read_csv(
        STOCK_LIST,
        dtype={"code":str}
    )


    stocks["code"]=(
        stocks["code"]
        .str.zfill(6)
    )



    last=get_progress()

    start=0


    if last:

        for i,row in stocks.iterrows():

            if row["code"]==last:

                start=i+1
                break




    total=len(stocks)


    for i,row in stocks.iloc[start:].iterrows():


        code=row["code"]


        print(
            f"[{i+1}/{total}] 更新:",
            code
        )


        try:


            text=get_price(code)


            price=parse_price(text)



            if price is None:

                print(
                    code,
                    "失败"
                )

                continue



            df=load_stock(code)


            df=update_stock(
                df,
                price
            )


            save_stock(
                code,
                df
            )


            save_progress(code)



            print(
                code,
                "完成",
                price
            )


        except Exception:


            log_error(
                traceback.format_exc()
            )


        time.sleep(
            random.uniform(
                0.5,
                1.5
            )
        )



    print("================")
    print(
        "行情更新完成"
    )



if __name__=="__main__":

    update()

'''



os.makedirs("ai",exist_ok=True)


with open(
    "ai/update_data.py",
    "w",
    encoding="utf-8"
) as f:

    f.write(content)



print(
    "update_data.py V2.3.1 已生成"
)