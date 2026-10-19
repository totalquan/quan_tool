"""
数据获取模块：从 AkShare 拉取 A 股日线行情
=================================================
原理：量化的第一步是"喂数据"。我们需要历史K线（OHLCV：开高低收量），
      才能回测策略。AkShare 是免费开源的 Python 财经数据接口，
      底层爬的是东方财富/新浪等公开数据，个人研究完全够用。

什么是 OHLCV？
  - Open（开盘价）：当天第一笔成交价
  - High（最高价）：当天最高成交价
  - Low（最低价）：当天最低成交价
  - Close（收盘价）：当天最后一笔成交价 —— 做策略最常用这个
  - Volume（成交量）：当天成交了多少股
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from app.database import SessionLocal, DailyBar


def fetch_daily_data(ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取某只股票的日线数据，先查数据库缓存，没有再从网上拉。

    参数：
        ts_code: 股票代码，如 "600519"（贵州茅台）
        start_date: 开始日期，格式 "20230101"
        end_date: 结束日期，格式 "20240101"

    返回：
        DataFrame，列：trade_date, open, high, low, close, volume, amount
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        # 默认往前取2年
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")

    db = SessionLocal()
    try:
        # 1. 先查数据库里有没有这段数据
        records = (
            db.query(DailyBar)
            .filter(
                DailyBar.ts_code == ts_code,
                DailyBar.trade_date >= datetime.strptime(start_date, "%Y%m%d").date(),
                DailyBar.trade_date <= datetime.strptime(end_date, "%Y%m%d").date(),
            )
            .order_by(DailyBar.trade_date)
            .all()
        )

        if records:
            # 数据库里有，直接拼 DataFrame 返回
            df = pd.DataFrame([
                {
                    "trade_date": r.trade_date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "amount": r.amount,
                }
                for r in records
            ])
            print(f"[缓存命中] {ts_code} 从数据库读取 {len(df)} 条记录")
            return df

        # 2. 数据库没有，从 AkShare 拉
        print(f"[网络获取] {ts_code} 从 AkShare 拉取 {start_date} ~ {end_date}")
        # akshare 的 stock_zh_a_hist 接口：A股历史行情
        # period="daily" 日线；adjust="qfq" 前复权（把历史价格按分红除权调整成可比的）
        raw = ak.stock_zh_a_hist(
            symbol=ts_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )

        if raw.empty:
            raise ValueError(f"未获取到 {ts_code} 的数据，请检查股票代码")

        # 统一列名（akshare 返回的是中文列名）
        df = pd.DataFrame({
            "trade_date": pd.to_datetime(raw["日期"]).dt.date,
            "open": raw["开盘"].astype(float),
            "high": raw["最高"].astype(float),
            "low": raw["最低"].astype(float),
            "close": raw["收盘"].astype(float),
            "volume": raw["成交量"].astype(float),
            "amount": raw["成交额"].astype(float),
        })

        # 3. 存入数据库缓存
        for _, row in df.iterrows():
            bar = DailyBar(
                ts_code=ts_code,
                trade_date=row["trade_date"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                amount=row["amount"],
            )
            db.merge(bar)  # merge = 有就更新，没有就插入
        db.commit()
        print(f"[缓存写入] 已存 {len(df)} 条到本地数据库")

        return df

    finally:
        db.close()
