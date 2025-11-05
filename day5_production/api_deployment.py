"""
Day 5: API 部署示例
演示如何将智能体部署为 API 端点（使用 FastAPI 和 Flask）
"""

import os
import time
from typing import Optional, Dict, Any
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# 加载环境变量
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 配置 Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

from model_utils import get_model_for_task, get_default_model

# 初始化 FastAPI 应用
app = FastAPI(
    title="AI Agent API",
    description="将智能体部署为 API 端点",
    version="1.0.0"
)

# 全局模型实例（在应用启动时初始化）
model = None

def initialize_model():
    """初始化模型"""
    global model
    if model is None:
        model_name = get_default_model()
        model = genai.GenerativeModel(model_name)
        print(f"模型已初始化: {model_name}")

# 请求模型
class AgentRequest(BaseModel):
    query: str
    task_type: Optional[str] = "general"
    max_tokens: Optional[int] = 1000

# 响应模型
class AgentResponse(BaseModel):
    answer: str
    model_used: str
    latency_ms: float
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    initialize_model()
    print("API 服务已启动")

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "AI Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "与智能体对话",
            "GET /health": "健康检查",
            "GET /models": "获取可用模型"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_initialized": model is not None
    }

@app.get("/models")
async def get_models():
    """获取可用模型信息"""
    return {
        "default_model": get_default_model(),
        "model_for_simple": get_model_for_task("simple"),
        "model_for_complex": get_model_for_task("complex")
    }

@app.post("/chat", response_model=AgentResponse)
async def chat(request: AgentRequest):
    """
    与智能体对话的端点
    
    Args:
        request: 包含查询和可选参数的请求体
        
    Returns:
        智能体的回答和元数据
    """
    if model is None:
        initialize_model()
    
    start_time = time.time()
    
    try:
        # 根据任务类型选择模型
        if request.task_type != "general":
            model_name = get_model_for_task(request.task_type)
            current_model = genai.GenerativeModel(model_name)
        else:
            current_model = model
            model_name = get_default_model()
        
        # 生成回答
        response = current_model.generate_content(
            request.query,
            generation_config={
                "max_output_tokens": request.max_tokens,
                "temperature": 0.7,
            }
        )
        
        # 计算延迟
        latency_ms = (time.time() - start_time) * 1000
        
        # 获取回答文本
        answer = response.text if response.text else "抱歉，无法生成回答。"
        
        return AgentResponse(
            answer=answer,
            model_used=model_name,
            latency_ms=round(latency_ms, 2),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时出错: {str(e)}"
        )

# Flask 版本的示例（可选）
def create_flask_app():
    """创建 Flask 应用（作为对比示例）"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask 未安装，跳过 Flask 示例")
        return None
    
    flask_app = Flask(__name__)
    
    @flask_app.route("/", methods=["GET"])
    def flask_root():
        return jsonify({
            "message": "AI Agent API (Flask)",
            "version": "1.0.0"
        })
    
    @flask_app.route("/chat", methods=["POST"])
    def flask_chat():
        data = request.get_json()
        query = data.get("query", "")
        task_type = data.get("task_type", "general")
        
        if not query:
            return jsonify({"error": "查询不能为空"}), 400
        
        start_time = time.time()
        
        try:
            model_name = get_model_for_task(task_type)
            current_model = genai.GenerativeModel(model_name)
            response = current_model.generate_content(query)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return jsonify({
                "answer": response.text,
                "model_used": model_name,
                "latency_ms": round(latency_ms, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return flask_app

if __name__ == "__main__":
    print("=" * 60)
    print("AI Agent API 部署示例")
    print("=" * 60)
    print("\n启动 FastAPI 服务器...")
    print("API 文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("\n使用示例:")
    print("  curl -X POST http://localhost:8000/chat \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"query\": \"你好，请介绍一下自己\"}'")
    print("\n按 Ctrl+C 停止服务器\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

