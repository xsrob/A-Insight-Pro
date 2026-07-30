import os
import pandas as pd


DATA_DIR="data"


def clean():

    print("================")
    print("清理实时行情污染")
    print("================")


    count=0


    for file in os.listdir(DATA_DIR):

        if not file.endswith(".csv"):
            continue


        path=os.path.join(
            DATA_DIR,
            file
        )


        try:

            df=pd.read_csv(path)


            if "date" not in df.columns:
                continue


            old=len(df)


            # 删除分钟级日期
            df=df[
                ~df["date"]
                .astype(str)
                .str.match(r"^\d{14}$")
            ]


            # 删除成交量为0的伪日K
            if "volume" in df.columns:

                df=df[
                    ~(
                        (df["volume"]==0)
                        &
                        (df["date"].astype(str).str.len()==10)
                    )
                ]


            new=len(df)


            if new != old:


                df.to_csv(
                    path,
                    index=False,
                    encoding="utf-8-sig"
                )


                count+=1

                print(
                    file,
                    old,
                    "->",
                    new
                )


        except Exception as e:

            print(
                file,
                e
            )


    print("================")
    print(
        "清理完成:",
        count,
        "个文件"
    )
    print("================")


if __name__=="__main__":

    clean()