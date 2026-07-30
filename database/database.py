"""
A-Insight Pro
数据库连接模块
"""

from sqlalchemy import create_engine
from config import settings


# 创建数据库连接

engine = create_engine(
    f"sqlite:///{settings.DATABASE_NAME}",
    echo=False
)


def get_engine():

    """
    返回数据库连接
    """

    return engine