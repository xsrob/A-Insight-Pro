"""
A-Insight Pro
股票行情采集 V6

数据源:
新浪历史K线接口

功能:
1. 自动股票池
2. 历史行情采集
3. CSV保存
4. 绕过系统代理
"""


import os
import time
import requests
import pandas as pd


DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)



# ==========================
# 股票池
# ==========================

def load_stock_list():


    file = os.path.join(
        DATA_DIR,
        "stock_list.csv"
    )


    # 如果存在读取

    if os.path.exists(file):

        df = pd.read_csv(
            file,
            dtype=str
        )


        stocks=[]


        for _,r in df.iterrows():

            stocks.append({

                "code":
                    str(r["code"]).zfill(6),

                "name":
                    r.get("name","")

            })


        return stocks



    # 不存在自动创建基础股票池

    print(
        "没有股票列表，自动生成"
    )


    codes = [

        "000001",
        "000002",
        "000006",
        "000007",
        "000008",
        "000009",
        "000010",
        "000011",
        "000012",
        "000014",

    ]


    stocks=[]


    for c in codes:

        stocks.append({

            "code":c,

            "name":""

        })


    return stocks





# ==========================
# 新浪行情
# ==========================


def get_stock_price(code):

    import json


    try:


        if code.startswith("6"):

            symbol = "sh" + code

        else:

            symbol = "sz" + code



        url = (
            "https://quotes.sina.cn/cn/api/jsonp_v2.php/"
            "var%20_data=/CN_MarketDataService.getKLineData"
        )


        params = {

            "symbol": symbol,

            "scale": 240,

            "ma": "no",

            "datalen": 1000

        }


        headers = {

            "User-Agent":
            "Mozilla/5.0"

        }



        s = requests.Session()


        # 禁用系统代理
        s.trust_env = False



        for retry in range(3):

            try:


                r = s.get(

                    url,

                    params=params,

                    headers=headers,

                    timeout=8

                )


                text = r.text



                start=text.find("[")

                end=text.rfind("]")



                if start < 0 or end < 0:

                    continue



                data=text[start:end+1]



                records=json.loads(data)



                if not records:

                    return None



                df=pd.DataFrame(records)



                if df.empty:

                    return None



                df.rename(

                    columns={

                        "day":"date"

                    },

                    inplace=True

                )



                df["date"]=pd.to_datetime(

                    df["date"]

                )



                for c in [

                    "open",

                    "high",

                    "low",

                    "close",

                    "volume"

                ]:


                    df[c]=pd.to_numeric(

                        df[c],

                        errors="coerce"

                    )



                df=df.dropna()



                return df



            except Exception as e:


                print(

                    code,

                    "第",

                    retry+1,

                    "次失败",

                    e

                )


                time.sleep(2)



        return None



    except Exception as e:


        print(

            code,

            "失败:",

            e

        )


        return None


    try:


        if code.startswith("6"):

            symbol="sh"+code

        else:

            symbol="sz"+code



        url=(

            "https://quotes.sina.cn/cn/api/jsonp_v2.php/"
            "var%20_data=/CN_MarketDataService.getKLineData"

        )



        params={


            "symbol":
                symbol,


            "scale":
                240,


            "ma":
                "no",


            "datalen":
                1000

        }



        headers={


            "User-Agent":

            "Mozilla/5.0"

        }



        s=requests.Session()


        # 关键:
        # 不读取系统代理

        s.trust_env=False



        r=s.get(

            url,

            params=params,

            headers=headers,

            timeout=15

        )



        text=r.text



        start=text.find("[")

        end=text.rfind("]")



        if start<0:

            print(
                code,
                "无数据"
            )

            return None



        json_data=text[start:end+1]



        df=pd.read_json(
            json_data
        )



        if df.empty:

            return None



        df.rename(

            columns={

                "day":"date"

            },

            inplace=True

        )



        df["date"]=pd.to_datetime(

            df["date"]

        )



        cols=[

            "open",

            "high",

            "low",

            "close",

            "volume"

        ]



        for c in cols:

            if c in df.columns:

                df[c]=pd.to_numeric(

                    df[c],

                    errors="coerce"

                )



        df=df.dropna()



        return df



    except Exception as e:


        print(

            code,

            "失败:",

            e

        )


        return None





# ==========================
# 保存
# ==========================


def save_stock_price(stock_list):


    print(

        "本次采集数量:",

        len(stock_list)

    )



    success=0



    for i,stock in enumerate(stock_list,1):


        code=stock["code"]



        print(

            f"{i}/{len(stock_list)}",

            code

        )


        df=get_stock_price(code)



        if df is None:

            continue



        file=os.path.join(

            DATA_DIR,

            f"{code}.csv"

        )



        df.to_csv(

            file,

            index=False,

            encoding="utf-8-sig"

        )


        success+=1



        print(

            code,

            "保存成功",

            len(df),

            "条"

        )


        time.sleep(1)



    print("================")

    print(

        "完成:",

        success,

        "/",

        len(stock_list)

    )





# ==========================


if __name__=="__main__":


    stocks=load_stock_list()


    if not stocks:

        print(
            "股票池为空"
        )

    else:

        save_stock_price(
            stocks
        )
    