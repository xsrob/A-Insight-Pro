from data_center.index_data import get_shanghai_index

from engines.market_engine import MarketEngine


# 获取指数数据

data = get_shanghai_index()


# 创建市场分析器

engine = MarketEngine()


# 分析

result = engine.analyze(data)


print("====================")

print("市场分析结果:")

print(result)

print("====================")