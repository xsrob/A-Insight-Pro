"""
A-Insight Pro
数据质量检查
"""

import os
import pandas as pd


DATA_DIR="data"



def check():


    files=[

        f for f in os.listdir(DATA_DIR)

        if f.endswith(".csv")

    ]


    print(
        "发现文件:",
        len(files)
    )


    total=0



    for f in files:


        path=os.path.join(
            DATA_DIR,
            f
        )


        try:


            df=pd.read_csv(path)



            print(
                f,
                "行数:",
                len(df),
                "字段:",
                list(df.columns)
            )



            if df.isnull().sum().sum()>0:

                print(
                    "⚠存在空值"
                )

            else:

                print(
                    "正常"
                )



            total+=len(df)



        except Exception as e:


            print(
                f,
                "错误:",
                e
            )



    print("================")

    print(
        "总数据量:",
        total
    )



if __name__=="__main__":

    check()