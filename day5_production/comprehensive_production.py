"""
Day 5: 综合生产示例
结合 API 部署、多智能体系统和扩展性优化的完整示例
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
from dotenv import load_dotenv

# 加载环境变量
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 配置 Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

from model_utils import get_model_for_task, get_default_model
from scalability_optimization import OptimizedAgent, ResponseCache, AsyncAgent
from multi_agent_system import MultiAgentSystem, PlannerAgent, ExecutorAgent

# 初始化 FastAPI 应用
app = FastAPI(
    title="Production AI Agent System",
    description="生产级 AI 智能体系统：API 部署 + 多智能体 + 优化",
    version="1.0.0"
)

# 全局实例
optimized_agent = None
multi_agent_system = None
async_agent = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化组件"""
    global optimized_agent, multi_agent_system, async_agent
    
    print("初始化生产级智能体系统...")
    optimized_agent = OptimizedAgent()
    multi_agent_system = MultiAgentSystem()
    async_agent = AsyncAgent()
    print("系统初始化完成")

# 请求模型
class SimpleRequest(BaseModel):
    query: str
    use_cache: bool = True

class BatchRequest(BaseModel):
    queries: List[str]
    use_cache: bool = True

class MultiAgentRequest(BaseModel):
    query: str
    use_planning: bool = True

# 响应模型
class AgentResponse(BaseModel):
    answer: str
    model_used: str
    latency_ms: float
    cached: bool = False
    timestamp: str

class BatchResponse(BaseModel):
    results: List[AgentResponse]
    total_time_ms: float
    average_latency_ms: float

class MultiAgentResponse(BaseModel):
    task_id: str
    original_query: str
    subtasks_count: int
    final_answer: str
    total_time_seconds: float
    status: str

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "Production AI Agent System",
        "version": "1.0.0",
        "features": [
            "优化智能体（缓存 + 模型选择）",
            "多智能体系统（规划 + 执行）",
            "异步批量处理",
            "API 端点部署"
        ],
        "endpoints": {
            "POST /chat": "单次对话（优化版）",
            "POST /chat/batch": "批量对话（异步）",
            "POST /chat/multi-agent": "多智能体任务处理",
            "GET /health": "健康检查",
            "GET /stats": "系统统计",
            "GET /cache/stats": "缓存统计"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "components": {
            "optimized_agent": optimized_agent is not None,
            "multi_agent_system": multi_agent_system is not None,
            "async_agent": async_agent is not None
        }
    }

@app.get("/stats")
async def get_stats():
    """获取系统统计"""
    cache_stats = optimized_agent.get_cache_stats() if optimized_agent else {}
    
    return {
        "cache": cache_stats,
        "models": {
            "simple": get_model_for_task("simple"),
            "complex": get_model_for_task("complex"),
            "default": get_default_model()
        }
    }

@app.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计"""
    if optimized_agent is None:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    return optimized_agent.get_cache_stats()

@app.post("/chat", response_model=AgentResponse)
async def chat(request: SimpleRequest):
    """
    单次对话端点（优化版）
    
    使用缓存和智能模型选择
    """
    if optimized_agent is None:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    start_time = time.time()
    
    try:
        result = optimized_agent.process(request.query, use_cache=request.use_cache)
        
        return AgentResponse(
            answer=result["answer"],
            model_used=result["model"],
            latency_ms=result["latency_ms"],
            cached=result.get("cached", False),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/batch", response_model=BatchResponse)
async def chat_batch(request: BatchRequest):
    """
    批量对话端点（异步处理）
    
    并发处理多个查询，提高吞吐量
    """
    if async_agent is None:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    start_time = time.time()
    
    try:
        # 异步批量处理
        results = await async_agent.process_batch(request.queries)
        
        total_time_ms = (time.time() - start_time) * 1000
        average_latency = sum(r["latency_ms"] for r in results) / len(results)
        
        # 转换为响应格式
        agent_responses = [
            AgentResponse(
                answer=r["answer"],
                model_used=r["model"],
                latency_ms=r["latency_ms"],
                cached=False,  # 异步处理暂不支持缓存
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            for r in results
        ]
        
        return BatchResponse(
            results=agent_responses,
            total_time_ms=round(total_time_ms, 2),
            average_latency_ms=round(average_latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/multi-agent", response_model=MultiAgentResponse)
async def chat_multi_agent(request: MultiAgentRequest):
    """
    多智能体任务处理端点
    
    使用规划智能体和执行智能体协作处理复杂任务
    """
    if multi_agent_system is None:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    try:
        # 在线程池中执行同步的多智能体处理
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: multi_agent_system.process_task(request.query)
        )
        
        return MultiAgentResponse(
            task_id=result["task_id"],
            original_query=result["original_query"],
            subtasks_count=result["subtasks_count"],
            final_answer=result["final_answer"],
            total_time_seconds=result["total_time_seconds"],
            status=result["status"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def demonstrate_usage():
    """演示使用示例"""
    print("=" * 60)
    print("生产级智能体系统使用示例")
    print("=" * 60)
    
    print("""
1. 启动服务器:
   python comprehensive_production.py

2. API 端点使用示例:

   a) 单次对话（优化版）:
   curl -X POST http://localhost:8000/chat \\
     -H 'Content-Type: application/json' \\
     -d '{"query": "什么是人工智能？", "use_cache": true}'

   b) 批量对话（异步）:
   curl -X POST http://localhost:8000/chat/batch \\
     -H 'Content-Type: application/json' \\
     -d '{"queries": ["什么是AI？", "什么是机器学习？"], "use_cache": false}'

   c) 多智能体任务:
   curl -X POST http://localhost:8000/chat/multi-agent \\
     -H 'Content-Type: application/json' \\
     -d '{"query": "请帮我计算 25×4，然后解释什么是AI", "use_planning": true}'

   d) 查看统计:
   curl http://localhost:8000/stats
   curl http://localhost:8000/cache/stats

3. 交互式 API 文档:
   访问 http://localhost:8000/docs 查看 Swagger UI
   访问 http://localhost:8000/redoc 查看 ReDoc
    """)

if __name__ == "__main__":
    print("=" * 60)
    print("生产级 AI 智能体系统")
    print("=" * 60)
    print("\n启动服务器...")
    print("API 文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("\n按 Ctrl+C 停止服务器\n")
    
    demonstrate_usage()
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

