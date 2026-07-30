"""
A-Insight Pro
预测偏差自动修正系统 V1.0
"""


import os
import json
import pandas as pd



LEARNING_FILE = "reports/ai_learning_feedback.csv"

OUTPUT_FILE = "reports/predict_adjust.json"




def adjust_learning():


    print("================")
    print("预测参数学习启动")
    print("================")



    if not os.path.exists(LEARNING_FILE):

        print(
            "没有学习数据"
        )

        return



    df = pd.read_csv(

        LEARNING_FILE,

        encoding="utf-8-sig"

    )



    if df.empty:

        print(
            "学习数据为空"
        )

        return




    avg_error = (

        df["avg_error"]

        .mean()

    )



    factor = 1.0



    # AI长期高估

    if avg_error < -5:

        factor = 0.85



    # AI长期低估

    elif avg_error > 5:

        factor = 1.15



    result = {


        "predict_factor":

            factor,


        "avg_error":

            round(
                avg_error,
                2
            ),


        "sample":

            len(df)


    }




    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            result,

            f,

            ensure_ascii=False,

            indent=4

        )




    print("================")

    print(
        "预测修正完成"
    )


    print(result)


    print("================")





if __name__=="__main__":

    adjust_learning()