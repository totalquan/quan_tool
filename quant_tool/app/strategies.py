"""
策略模块：定义交易信号
=================================================
原理：一个量化策略本质上就是一个"函数"——
      输入：历史行情数据（DataFrame）
      输出：每天的持仓信号（1=持有/买入，0=空仓/卖出）

      回测引擎拿到这个信号序列后，模拟"按信号买卖"能赚多少钱。

核心概念：
  - 均线（MA）：过去 N 天收盘价的平均值，用来判断趋势方向
  - 金叉：短期均线上穿长期均线 → 趋势转多 → 买入信号
  - 死叉：短期均线下穿长期均线 → 趋势转空 → 卖出信号
"""

import pandas as pd
import numpy as np


def ma_crossover(df: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """
    双均线交叉策略（最经典的入门策略）
    ====================================
    逻辑：
      1. 算短期均线 MA5（最近5天平均收盘价）
      2. 算长期均线 MA20（最近20天平均收盘价）
      3. 当 MA5 上穿 MA20（金叉）→ 第二天开盘买入并持有
      4. 当 MA5 下穿 MA20（死叉）→ 第二天开盘卖出空仓

    为什么用均线？
      短期均线代表"最近大家的平均持仓成本"，长期均线代表"中期趋势"。
      短期站上长期，说明最近买盘强于中期平均，趋势可能向上。

    参数：
        short_window: 短期均线天数，默认5
        long_window:  长期均线天数，默认20

    返回：
        原 DataFrame 新增列：ma_short, ma_long, signal（每天的目标仓位 0或1）
    """
    df = df.copy()

    # rolling(window=N).mean() = 取最近N行算平均，就是均线
    df["ma_short"] = df["close"].rolling(window=short_window).mean()
    df["ma_long"] = df["close"].rolling(window=long_window).mean()

    # signal = 1 表示当天应该持有，0 表示空仓
    # 规则：短期均线 > 长期均线 就持有，否则空仓
    df["signal"] = 0
    df.loc[df["ma_short"] > df["ma_long"], "signal"] = 1

    return df


def rsi_strategy(df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.DataFrame:
    """
    RSI 相对强弱指标策略（均值回归思路）
    ====================================
    RSI 衡量最近N天里，涨的幅度占总波动幅度的百分比，范围 0~100：
      - RSI < 30（超卖区）：跌过头了，可能反弹 → 买入
      - RSI > 70（超买区）：涨过头了，可能回调 → 卖出

    和双均线的区别：
      双均线是"趋势跟踪"（涨了追、跌了跑）
      RSI 是"均值回归"（跌多了买、涨多了卖）—— 两个思路完全相反
    """
    df = df.copy()

    # 算每天的涨跌
    delta = df["close"].diff()

    # 把上涨和下跌分开
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # N天平均涨跌幅
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # RS = 平均涨幅 / 平均跌幅；RSI = 100 - 100/(1+RS)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 信号：RSI < 30 买入持有，RSI > 70 卖出空仓
    # 注意：这里用"持仓状态"而不是"单日信号"——RSI在30以下时持续持有
    df["signal"] = 0
    holding = False
    for i in range(len(df)):
        r = df["rsi"].iloc[i]
        if pd.isna(r):
            continue
        if r < oversold:
            holding = True   # 超卖 → 买入持有
        elif r > overbought:
            holding = False  # 超买 → 卖出空仓
        df.iloc[i, df.columns.get_loc("signal")] = 1 if holding else 0

    return df


# 策略注册表：前端传 strategy_name 来选策略
STRATEGIES = {
    "ma_crossover": {
        "name": "双均线交叉",
        "func": ma_crossover,
        "params": {"short_window": 5, "long_window": 20},
        "description": "趋势跟踪：短期均线上穿长期均线买入，下穿卖出",
    },
    "rsi": {
        "name": "RSI均值回归",
        "func": rsi_strategy,
        "params": {"period": 14, "oversold": 30, "overbought": 70},
        "description": "逆向操作：RSI低于30超卖买入，高于70超买卖出",
    },
}
