from data_center.index_data import get_shanghai_index

from engines.market_engine import MarketEngine

from engines.risk_engine import RiskEngine



data = get_shanghai_index()


market = MarketEngine()

market_result = market.analyze(data)



risk = RiskEngine()

risk_result = risk.analyze(
    market_result
)


print("====================")

print("市场结果:")

print(market_result)


print("风险分析:")

print(risk_result)


print("====================")