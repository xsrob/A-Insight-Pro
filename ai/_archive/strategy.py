"""
A-Insight Pro
交易策略层
"""


import pandas as pd
import os


INPUT = "reports/ai_prediction.csv"

OUTPUT = "reports/trade_pool.csv"



def strategy():


    df = pd.read_csv(
        INPUT
    )


    result=[]


    for _,row in df.iterrows():


        predict = row["predict_percent"]



        # 策略规则

        if predict >= 5:

            signal="BUY"


        elif predict >= 2:

            signal="WATCH"


        else:

            signal="HOLD"



        # 风险等级

        if predict >=10:

            risk="HIGH"


        elif predict>=5:

            risk="MEDIUM"


        else:

            risk="LOW"



        result.append({

            "rank":
            row["rank"],


            "code":
            row["code"],


            "predict_percent":
            predict,


            "signal":
            signal,


            "risk":
            risk

        })



    out=pd.DataFrame(
        result
    )


    out.to_csv(

        OUTPUT,

        index=False,

        encoding="utf-8-sig"

    )


    print("================")

    print("策略生成完成")

    print(
        OUTPUT
    )

    print()

    print(
        out.head(20)
    )




if __name__=="__main__":

    strategy()