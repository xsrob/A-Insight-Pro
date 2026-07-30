"""
A-Insight Pro
数据存储模块
"""

from database.database import get_engine
from database.models import Base, IndexData, StockList

from data_center.index_data import get_shanghai_index
from data_center.stock_data import get_stock_list


from sqlalchemy.orm import sessionmaker



def create_tables():

    """
    创建数据库表
    """

    engine = get_engine()

    Base.metadata.create_all(engine)



def save_index_data():

    """
    保存指数数据
    """

    print("正在获取指数数据...")

    data = get_shanghai_index()


    engine = get_engine()

    Session = sessionmaker(
        bind=engine
    )

    session = Session()


    latest = data.tail(1).iloc[0]


    index = IndexData(

        date=str(latest["date"]),

        close=float(latest["close"]),

        volume=float(latest["volume"])

    )


    session.add(index)

    session.commit()

    session.close()


    print("指数数据保存完成")




def save_stock_list():

    """
    保存股票列表
    """

    print("正在获取股票列表...")


    stocks = get_stock_list()


    if stocks is None:

        print("股票数据获取失败")

        return


    engine = get_engine()


    Session = sessionmaker(
        bind=engine
    )


    session = Session()


    for _, row in stocks.iterrows():

        stock = StockList(

            code=row["code"],

            name=row["name"]

        )

        session.add(stock)


    session.commit()

    session.close()


    print(
        "股票列表保存完成:",
        len(stocks)
    )




if __name__ == "__main__":


    print("开始初始化数据...")


    create_tables()


    save_index_data()


    save_stock_list()


    print("全部完成")