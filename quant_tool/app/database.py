"""
数据库模块：用 SQLite 存储历史行情和回测结果
=================================================
原理：量化系统需要把"下载下来的行情数据"和"回测跑出来的结果"存起来，
      下次不用重复下载、重复计算。这里用 SQLite（一个文件就是一个数据库），
      不需要安装任何数据库软件，开箱即用。
"""

from sqlalchemy import create_engine, Column, String, Float, Date, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# 创建数据库引擎：sqlite:///./quant.db 表示在当前目录下建一个 quant.db 文件
engine = create_engine("sqlite:///./quant.db", echo=False)

# 会话工厂：每次需要操作数据库时，用 Session() 拿一个会话
SessionLocal = sessionmaker(bind=engine)

# ORM 基类：所有数据表模型都继承它
Base = declarative_base()


class DailyBar(Base):
    """日线行情表：存一只股票每天的 OHLCV 数据"""
    __tablename__ = "daily_bars"

    # 复合主键：股票代码 + 日期（同一只股票同一天只有一条记录）
    ts_code = Column(String(20), primary_key=True)   # 股票代码，如 "600519"
    trade_date = Column(Date, primary_key=True)      # 交易日期
    open = Column(Float)                              # 开盘价
    high = Column(Float)                              # 最高价
    low = Column(Float)                               # 最低价
    close = Column(Float)                             # 收盘价
    volume = Column(Float)                            # 成交量（手）
    amount = Column(Float)                            # 成交额（元）


class BacktestResult(Base):
    """回测结果表：每次跑完回测存一条记录"""
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50))                # 策略名，如 "双均线"
    ts_code = Column(String(20))                      # 回测的股票代码
    start_date = Column(Date)                         # 回测开始日期
    end_date = Column(Date)                           # 回测结束日期
    total_return = Column(Float)                      # 总收益率
    annual_return = Column(Float)                      # 年化收益率
    max_drawdown = Column(Float)                      # 最大回撤
    sharpe_ratio = Column(Float)                      # 夏普比率
    trade_count = Column(Integer)                      # 交易次数
    created_at = Column(DateTime, default=datetime.now)  # 创建时间
    detail = Column(Text)                             # 详细 JSON（收益曲线等）


# 启动时自动建表（如果表不存在）
Base.metadata.create_all(engine)
