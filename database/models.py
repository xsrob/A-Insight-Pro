"""
A-Insight Pro
数据库模型
"""

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float


Base = declarative_base()


class IndexData(Base):

    __tablename__ = "index_data"

    id = Column(
        Integer,
        primary_key=True
    )

    date = Column(String)

    open = Column(Float)

    high = Column(Float)

    low = Column(Float)

    close = Column(Float)

    volume = Column(Float)



class StockList(Base):

    __tablename__ = "stock_list"

    id = Column(
        Integer,
        primary_key=True
    )

    code = Column(String)

    name = Column(String)



class StockPrice(Base):

    __tablename__ = "stock_price"

    id = Column(
        Integer,
        primary_key=True
    )

    code = Column(String)

    date = Column(String)

    open = Column(Float)

    high = Column(Float)

    low = Column(Float)

    close = Column(Float)

    volume = Column(Float)