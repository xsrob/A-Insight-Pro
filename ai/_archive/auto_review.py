"""
A-Insight Pro
AI自动复盘系统 V1.0

功能:
1. 读取历史预测
2. 获取当前股票结果
3. 计算实际收益
4. 生成simulate_review.csv
"""

import os
import pandas as pd


PREDICT_HISTORY = "reports/prediction_history.csv"

DATA_DIR = "data"

OUTPUT = "reports/simulate_review.csv"



def load_predict():

    if not os.path.exists(PREDICT_HISTORY):

        print(
            "没有预测历史"
        )

        return None


    df = pd.read_csv(
        PREDICT_HISTORY,
        encoding="utf-8-sig"
    )


    return df



def get_latest_return(code):

    file = os.path.join(
        DATA_DIR,
        str(code).zfill(6)+".csv"
    )


    if not os.path.exists(file):

        return None



    try:

        df=pd.read_csv(
            file
        )


        if len(df)<2:

            return None


        close_now=float(
            df.iloc[-1]["close"]
        )


        close_before=float(
            df.iloc[-2]["close"]
        )


        return round(
            (close_now-close_before)
            /
            close_before
            *
            100,
            2
        )


    except:

        return None




def review():

    print("================")
    print(
        "AI自动复盘启动"
    )
    print("================")



    history=load_predict()


    if history is None:

        return



    results=[]



    for _,row in history.iterrows():


        code=str(row["code"]).zfill(6)


        actual=get_latest_return(
            code
        )


        if actual is None:

            continue



        predict=float(
            row["predict_percent"]
        )


        if actual>0:

            result="成功"

        else:

            result="失败"



        results.append(

            {

                "date":
                row["date"],

                "code":
                code,

                "predict_percent":
                predict,

                "actual_return":
                actual,

                "result":
                result

            }

        )



    if len(results)==0:

        print(
            "暂无可复盘交易"
        )

        return



    df=pd.DataFrame(
        results
    )


    os.makedirs(
        "reports",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )



    print("================")

    print(
        "复盘完成"
    )

    print(
        OUTPUT
    )

    print(
        df.head(20)
    )



if __name__=="__main__":

    review()