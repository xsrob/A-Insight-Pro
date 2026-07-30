"""
A-Insight Pro
个股分析引擎测试
"""


from sqlalchemy.orm import Session

from database.database import get_engine
from database.models import StockPrice

from engines.stock_engine import StockEngine



# 创建数据库连接

engine = get_engine()

session = Session(engine)



# 获取平安银行行情

data = session.query(
    StockPrice
).filter(
    StockPrice.code == "000001"
).order_by(
    StockPrice.date
).all()



# 转换DataFrame

import pandas as pd


df = pd.DataFrame(
    [
        {
            "date": x.date,
            "open": x.open,
            "high": x.high,
            "low": x.low,
            "close": x.close,
            "volume": x.volume
        }
        for x in data
    ]
)



print("====================")

print(
    "股票:",
    "000001 平安银行"
)


print(
    "行情数量:",
    len(df)
)



# 调用分析引擎

engine = StockEngine()


result = engine.analyze(
    df
)



print("====================")

print(
    "分析结果:"
)


print(result)


print("====================")


session.close()