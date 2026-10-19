"""
回测引擎：模拟"按信号买卖"能赚多少钱
=================================================
原理：回测 = 用历史数据模拟跑一遍策略，看如果当时按这个策略操作，
      最后是赚是亏。注意：历史业绩不代表未来！

核心逻辑（极简版）：
  - 每天的 signal=1 → 满仓持有这只股票
  - 每天的 signal=0 → 空仓（钱放着不动）
  - 当天收益率 = 当天涨跌幅 × 前一天的仓位
  - 累计收益率 = 每天收益率连乘

关键指标解释：
  - 总收益率：整个回测期间赚了百分之多少
  - 年化收益率：换算成"每年赚多少"，方便不同周期对比
  - 最大回撤：从最高点跌到最低点的最大跌幅（衡量"最惨的时候亏多少"）
  - 夏普比率：每承担1份风险能赚多少（>1算不错，>2算优秀）
  - 交易次数：一共买卖了几次
"""

import pandas as pd
import numpy as np
from app.strategies import STRATEGIES


def run_backtest(
    df: pd.DataFrame,
    strategy_key: str,
    initial_capital: float = 100000.0,
    commission_rate: float = 0.0003,  # 万三手续费，券商常见费率
) -> dict:
    """
    执行回测

    参数：
        df: 日线数据，必须包含 close 列
        strategy_key: 策略注册表的 key，如 "ma_crossover"
        initial_capital: 初始资金，默认10万
        commission_rate: 手续费率，每次买卖收万三

    返回：
        dict，包含各项指标和逐日收益曲线
    """
    strategy_info = STRATEGIES[strategy_key]
    strategy_func = strategy_info["func"]

    # 1. 跑策略，得到带 signal 列的 DataFrame
    df = strategy_func(df, **strategy_info["params"])

    # 2. 算每天的涨跌幅（今天收盘价 / 昨天收盘价 - 1）
    df["pct_change"] = df["close"].pct_change().fillna(0)

    # 3. 关键细节：signal 是"当天收盘后算出来的"，
    #    实际买卖要"第二天"才能执行，所以用 shift(1) 把信号往后挪一天
    #    否则就变成"用未来数据炒股"了，这是新手最容易犯的错！
    df["position"] = df["signal"].shift(1).fillna(0)

    # 4. 算每天策略收益率 = 持仓仓位 × 当天涨跌幅
    df["strategy_return"] = df["position"] * df["pct_change"]

    # 5. 扣手续费：仓位发生变化的那天（买入或卖出），收一次手续费
    df["trade"] = df["position"].diff().abs().fillna(0)  # 仓位变化量，第一天diff是NaN要填0
    df["commission"] = df["trade"] * commission_rate
    df["strategy_return"] = df["strategy_return"] - df["commission"]

    # 6. 算累计净值曲线（从1开始复利增长）
    df["cum_return"] = (1 + df["strategy_return"]).cumprod()
    # 基准：一直持有不动的收益曲线
    df["benchmark_return"] = (1 + df["pct_change"]).cumprod()

    # 7. 计算各项评价指标
    total_days = len(df)
    total_return = df["cum_return"].iloc[-1] - 1
    benchmark_total_return = df["benchmark_return"].iloc[-1] - 1

    # 年化收益：假设一年252个交易日
    annual_return = (1 + total_return) ** (252 / max(total_days, 1)) - 1

    # 最大回撤：累计净值从历史最高点跌下来的最大幅度
    running_max = df["cum_return"].cummax()
    drawdown = (df["cum_return"] - running_max) / running_max
    max_drawdown = drawdown.min()

    # 夏普比率：(年化收益 - 无风险利率) / 年化波动
    # 无风险利率粗略取 2%（一年期国债）
    daily_std = df["strategy_return"].std()
    sharpe = (annual_return - 0.02) / (daily_std * np.sqrt(252)) if daily_std > 0 else 0

    # 交易次数：仓位变化的次数（diff != 0 的天数）
    trade_count = int((df["trade"] > 0).sum())

    # 8. 整理返回结果（必须全部转成 Python 原生类型，否则 JSON 序列化会崩）
    #    numpy.float64 / numpy.int64 不是 Python 原生类型，FastAPI 序列化时会报错
    result = {
        "strategy_name": strategy_info["name"],
        "strategy_description": strategy_info["description"],
        "ts_code": str(df["ts_code"].iloc[0]) if "ts_code" in df.columns else "",
        "metrics": {
            "total_return": float(round(total_return * 100, 2)),
            "benchmark_return": float(round(benchmark_total_return * 100, 2)),
            "annual_return": float(round(annual_return * 100, 2)),
            "max_drawdown": float(round(max_drawdown * 100, 2)),
            "sharpe_ratio": float(round(sharpe, 3)),
            "trade_count": int(trade_count),
            "total_days": int(total_days),
        },
        # 收益曲线（给前端画图用）
        "equity_curve": [
            {
                "date": str(row["trade_date"]),
                "strategy": float(round(row["cum_return"], 4)),
                "benchmark": float(round(row["benchmark_return"], 4)),
            }
            for _, row in df.iterrows()
        ],
        # K线 + 买卖点（前端画图用）
        "kline": [
            {
                "date": str(row["trade_date"]),
                "open": float(round(row["open"], 2)),
                "close": float(round(row["close"], 2)),
                "low": float(round(row["low"], 2)),
                "high": float(round(row["high"], 2)),
                "signal": int(row["signal"]) if not pd.isna(row["signal"]) else 0,
            }
            for _, row in df.iterrows()
        ],
    }

    return result
