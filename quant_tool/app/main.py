"""
FastAPI 主入口：把回测能力暴露成 HTTP 接口
=================================================
启动后：
  - 前端页面：http://localhost:8000/
  - API 文档：http://localhost:8000/docs
  - 回测接口：POST /api/backtest
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from app.data_fetcher import fetch_daily_data
from app.backtester import run_backtest
from app.strategies import STRATEGIES
from app.database import SessionLocal, BacktestResult
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd

app = FastAPI(title="量化回测工具", version="1.0")

# 把 static 目录挂载成静态文件服务
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class BacktestRequest(BaseModel):
    """回测请求的参数格式"""
    ts_code: str = "600519"          # 默认贵州茅台
    strategy: str = "ma_crossover"     # 默认双均线
    start_date: str = "20230101"
    end_date: str = "20240101"


@app.get("/")
def index():
    """首页返回前端页面"""
    return FileResponse("app/static/index.html")


@app.get("/api/strategies")
def list_strategies():
    """列出所有可用策略"""
    return {
        "strategies": [
            {"key": k, "name": v["name"], "description": v["description"], "params": v["params"]}
            for k, v in STRATEGIES.items()
        ]
    }


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    """
    执行回测：
      1. 拉数据
      2. 跑策略
      3. 算指标
      4. 存数据库
      5. 返回给前端画图
    """
    try:
        # 拉数据
        df = fetch_daily_data(req.ts_code, req.start_date, req.end_date)
        df["ts_code"] = req.ts_code

        # 跑回测
        result = run_backtest(df, req.strategy)

        # 存数据库
        db = SessionLocal()
        record = BacktestResult(
            strategy_name=STRATEGIES[req.strategy]["name"],
            ts_code=req.ts_code,
            start_date=datetime.strptime(req.start_date, "%Y%m%d").date(),
            end_date=datetime.strptime(req.end_date, "%Y%m%d").date(),
            total_return=result["metrics"]["total_return"],
            annual_return=result["metrics"]["annual_return"],
            max_drawdown=result["metrics"]["max_drawdown"],
            sharpe_ratio=result["metrics"]["sharpe_ratio"],
            trade_count=result["metrics"]["trade_count"],
        )
        db.add(record)
        db.commit()
        db.close()

        return {"code": 0, "data": result}

    except Exception as e:
        return {"code": -1, "msg": str(e)}


@app.get("/api/history")
def history():
    """查看历史回测记录"""
    db = SessionLocal()
    records = db.query(BacktestResult).order_by(BacktestResult.created_at.desc()).limit(20).all()
    db.close()
    return {
        "records": [
            {
                "id": r.id,
                "strategy": r.strategy_name,
                "stock": r.ts_code,
                "period": f"{r.start_date} ~ {r.end_date}",
                "total_return": r.total_return,
                "sharpe": r.sharpe_ratio,
                "created_at": str(r.created_at),
            }
            for r in records
        ]
    }


@app.post("/api/demo")
def demo(req: BacktestRequest):
    """
    演示模式：用模拟数据跑回测，不需要联网。
    用来验证工具是否正常工作。
    """
    try:
        # 生成一段有波动的模拟K线（先涨后跌再涨，方便看到买卖信号）
        np.random.seed(42)
        n = 250
        end = date.today()
        dates = [end - timedelta(days=n-i) for i in range(n)]

        # 构造一个"先涨后跌再涨"的价格序列
        trend = np.concatenate([
            np.linspace(0, 20, 80),    # 前80天上涨
            np.linspace(20, -5, 90),   # 中间90天下跌
            np.linspace(-5, 15, 80),   # 后80天上涨
        ])
        noise = np.random.randn(n) * 1.5
        close_prices = 100 + trend + noise

        df = pd.DataFrame({
            "trade_date": dates,
            "open": close_prices + np.random.randn(n) * 0.5,
            "high": close_prices + abs(np.random.randn(n) * 1.0),
            "low": close_prices - abs(np.random.randn(n) * 1.0),
            "close": close_prices,
            "volume": np.random.rand(n) * 1e6,
            "amount": np.random.rand(n) * 1e8,
            "ts_code": "DEMO",
        })

        result = run_backtest(df, req.strategy)
        result["ts_code"] = "DEMO（模拟数据）"
        return {"code": 0, "data": result}

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=200,
            content={"code": -1, "msg": f"{type(e).__name__}: {str(e)}", "detail": traceback.format_exc()}
        )
