# Day 5: 从原型到生产（Prototype to Production）

## 学习目标
- 学习将智能体部署为 API 端点的最佳实践（使用 FastAPI 和 Flask）
- 掌握多智能体系统（MAS）和 Agent2Agent (A2A) 协议
- 了解扩展性和成本优化策略（缓存、异步处理、模型选择）

## 文件说明
- `api_deployment.py`: API 部署示例 - 使用 FastAPI 将智能体部署为 API 端点
- `multi_agent_system.py`: 多智能体系统示例 - Planner 和 Executor 智能体协作
- `scalability_optimization.py`: 扩展性与成本优化示例 - 缓存、异步处理、模型选择
- `comprehensive_production.py`: 综合生产示例 - 结合所有概念的完整系统

## 环境设置

1. 安装依赖：
```bash
cd day5_production
pip install -r requirements.txt
```

2. 配置 API Key：
```bash
# 在项目根目录（google_AI_day）创建 .env 文件（如果还没有）
# 填入你的 GEMINI_API_KEY:
# GEMINI_API_KEY=your_api_key_here
```

3. 运行示例：

**方式一：从 day5_production 目录运行**
```bash
cd day5_production
python3 api_deployment.py              # API 部署示例（启动服务器）
python3 multi_agent_system.py          # 多智能体系统示例
python3 scalability_optimization.py    # 扩展性与成本优化示例
python3 comprehensive_production.py    # 综合生产示例（启动服务器）
```

**方式二：从项目根目录运行**
```bash
# 从项目根目录（google_AI_day）
python3 day5_production/api_deployment.py
python3 day5_production/multi_agent_system.py
python3 day5_production/scalability_optimization.py
python3 day5_production/comprehensive_production.py
```

**注意**: 
- `.env` 文件应位于项目根目录（`google_AI_day/.env`），代码会自动从父目录加载环境变量。
- 代码会自动检测可用的 Gemini 模型（优先使用 `gemini-1.5-flash` 或 `gemini-1.5-pro`）。
- 在 macOS 上，如果 `python` 命令不可用，请使用 `python3`。

## 核心概念

### 1. API 部署（API Deployment）

**概念**：
- 将智能体部署为 RESTful API 端点
- 使用 FastAPI 或 Flask 框架
- 支持健康检查、模型信息查询等端点
- 处理错误和异常情况

**实现**：
```python
app = FastAPI(title="AI Agent API")

@app.post("/chat")
async def chat(request: AgentRequest):
    response = model.generate_content(request.query)
    return AgentResponse(
        answer=response.text,
        model_used=model_name,
        latency_ms=latency_ms
    )
```

**关键端点**：
- `POST /chat`: 与智能体对话
- `GET /health`: 健康检查
- `GET /models`: 获取可用模型信息

**部署选项**：
- **本地开发**: 使用 `uvicorn` 或 `flask run`
- **生产环境**: 部署到 Cloud Functions、Cloud Run 或 Kubernetes
- **容器化**: 使用 Docker 打包应用

### 2. 多智能体系统（Multi-Agent System, MAS）

**概念**：
- **Agent2Agent (A2A) 协议**: 智能体之间的通信协议
- **规划智能体 (Planner Agent)**: 接收任务并分解为子任务
- **执行智能体 (Executor Agent)**: 接收子任务并使用工具执行

**架构**：
```
用户查询
    ↓
规划智能体 (Planner)
    ↓
子任务列表
    ↓
执行智能体 (Executor)
    ↓
工具执行结果
    ↓
最终回答
```

**实现**：
```python
# 规划智能体
planner = PlannerAgent()
task_plan = planner.plan("请帮我计算 25×4，然后查询天气")

# 执行智能体
executor = ExecutorAgent()
for subtask in task_plan.subtasks:
    result = executor.execute(subtask)

# 多智能体系统
mas = MultiAgentSystem()
result = mas.process_task("复杂任务")
```

**工作流程**：
1. **规划阶段**: Planner 分析任务，分解为子任务
2. **执行阶段**: Executor 逐个执行子任务
3. **汇总阶段**: 汇总执行结果，生成最终回答

### 3. 扩展性与成本优化（Scalability & Cost Optimization）

**优化策略**：

#### 3.1 缓存策略（Caching）

**概念**：
- 缓存常见查询的响应
- 减少重复的 API 调用
- 显著降低成本和延迟

**实现**：
```python
cache = ResponseCache(max_size=100, ttl=3600)
cached_response = cache.get(query, model_name)
if cached_response:
    return cached_response
else:
    response = generate_response(query)
    cache.set(query, model_name, response)
```

**缓存类型**：
- **TTL 缓存**: 基于时间的缓存（Time To Live）
- **LRU 缓存**: 最近最少使用缓存（Least Recently Used）
- **查询键**: 使用 MD5 哈希生成缓存键

#### 3.2 异步处理（Async Processing）

**概念**：
- 并发处理多个请求
- 提高吞吐量
- 减少总体响应时间

**实现**：
```python
async def process_async(query: str):
    response = await asyncio.run_in_executor(
        None,
        lambda: model.generate_content(query)
    )
    return response

# 批量处理
results = await asyncio.gather(
    *[process_async(q) for q in queries]
)
```

**优势**：
- **并发性**: 同时处理多个请求
- **效率**: 减少等待时间
- **扩展性**: 更好地利用系统资源

#### 3.3 模型选择策略（Model Selection）

**概念**：
- 根据任务复杂度选择合适的模型
- 简单任务使用 Flash 模型（更快、更便宜）
- 复杂任务使用 Pro 模型（更准确、更强大）

**实现**：
```python
def get_model_for_task(task_type: str) -> str:
    if task_type == "simple":
        return "gemini-1.5-flash"  # 更快、更便宜
    elif task_type == "complex":
        return "gemini-1.5-pro"    # 更准确、更强大
    else:
        return "gemini-1.5-flash"  # 默认

# 自动选择
use_complex = should_use_complex_model(query)
model = complex_model if use_complex else simple_model
```

**选择标准**：
- **简单查询**: 使用 Flash 模型
- **复杂查询**: 使用 Pro 模型（包含"分析"、"解释"、"比较"等关键词）

## 技术实现细节

### API 部署

**FastAPI 应用结构**：
```python
app = FastAPI(
    title="AI Agent API",
    description="将智能体部署为 API 端点",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    # 初始化模型
    initialize_model()

@app.post("/chat")
async def chat(request: AgentRequest):
    # 处理请求
    response = model.generate_content(request.query)
    return AgentResponse(...)
```

**请求/响应模型**：
- 使用 Pydantic 定义数据模型
- 自动验证和序列化
- 生成 API 文档

### 多智能体系统

**Planner Agent**：
- 使用 Pro 模型（更强大的推理能力）
- 将任务分解为 JSON 格式的子任务
- 每个子任务包含：描述、工具、参数

**Executor Agent**：
- 使用 Flash 模型（更快的执行速度）
- 支持多种工具：calculate, search, weather, calendar
- 执行子任务并返回结果

**MultiAgentSystem**：
- 协调 Planner 和 Executor
- 管理任务生命周期
- 生成最终回答

### 优化策略

**ResponseCache**：
- 使用 `cachetools.TTLCache` 实现
- 支持最大缓存大小和 TTL
- 跟踪缓存命中率

**AsyncAgent**：
- 使用 `asyncio` 实现异步处理
- 在线程池中执行同步 API 调用
- 支持批量处理

**OptimizedAgent**：
- 结合缓存、模型选择和异步处理
- 自动选择合适的模型
- 提供完整的优化统计

## 最佳实践

### 1. API 部署
- ✅ 使用 FastAPI 或 Flask 框架
- ✅ 实现健康检查端点
- ✅ 添加错误处理和日志记录
- ✅ 使用环境变量管理配置
- ✅ 实现请求限流和身份验证
- ✅ 使用容器化部署（Docker）

### 2. 多智能体系统
- ✅ 清晰的智能体职责划分
- ✅ 标准化的通信协议（A2A）
- ✅ 错误处理和重试机制
- ✅ 任务状态跟踪
- ✅ 支持任务并行执行

### 3. 扩展性与成本优化
- ✅ 实现响应缓存
- ✅ 使用异步处理提高吞吐量
- ✅ 根据任务复杂度选择模型
- ✅ 监控 API 调用和成本
- ✅ 实现请求批处理
- ✅ 使用延迟加载减少启动时间

### 4. 生产环境部署
- ✅ 使用容器化（Docker）
- ✅ 配置环境变量
- ✅ 实现日志和监控
- ✅ 设置健康检查
- ✅ 配置自动扩缩容
- ✅ 实现负载均衡

## 使用示例

### API 部署示例

**启动服务器**：
```bash
python3 api_deployment.py
```

**使用 curl 测试**：
```bash
# 单次对话
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "什么是人工智能？", "task_type": "general"}'

# 健康检查
curl http://localhost:8000/health

# 获取模型信息
curl http://localhost:8000/models
```

**访问 API 文档**：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 多智能体系统示例

```python
from multi_agent_system import MultiAgentSystem

mas = MultiAgentSystem()
result = mas.process_task("请帮我计算 25×4，然后解释什么是AI")

print(f"任务ID: {result['task_id']}")
print(f"子任务数: {result['subtasks_count']}")
print(f"最终回答: {result['final_answer']}")
```

### 扩展性优化示例

```python
from scalability_optimization import OptimizedAgent

agent = OptimizedAgent()

# 使用缓存
result = agent.process("什么是AI？", use_cache=True)
print(f"回答: {result['answer']}")
print(f"缓存命中: {result['cached']}")
print(f"模型: {result['model']}")

# 查看缓存统计
stats = agent.get_cache_stats()
print(f"缓存命中率: {stats['hit_rate']}%")
```

### 综合生产示例

**启动综合系统**：
```bash
python3 comprehensive_production.py
```

**API 端点**：
- `POST /chat`: 单次对话（优化版）
- `POST /chat/batch`: 批量对话（异步）
- `POST /chat/multi-agent`: 多智能体任务处理
- `GET /stats`: 系统统计
- `GET /cache/stats`: 缓存统计

## 部署到生产环境

### 使用 Docker

**创建 Dockerfile**：
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "comprehensive_production:app", "--host", "0.0.0.0", "--port", "8000"]
```

**构建和运行**：
```bash
docker build -t ai-agent-api .
docker run -p 8000:8000 --env-file .env ai-agent-api
```

### 部署到 Cloud Run

```bash
# 构建镜像
gcloud builds submit --tag gcr.io/PROJECT_ID/ai-agent-api

# 部署到 Cloud Run
gcloud run deploy ai-agent-api \
  --image gcr.io/PROJECT_ID/ai-agent-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key_here
```

## 测试状态

所有程序已通过测试，可以正常运行：

- ✅ `api_deployment.py` - API 部署示例
- ✅ `multi_agent_system.py` - 多智能体系统示例
- ✅ `scalability_optimization.py` - 扩展性与成本优化示例
- ✅ `comprehensive_production.py` - 综合生产示例

### 已知问题

- API 服务器需要手动停止（Ctrl+C）
- 异步处理需要 Python 3.7+
- 缓存统计需要多次请求才能看到效果

## 扩展学习

1. **高级部署**：
   - Kubernetes 部署
   - 负载均衡配置
   - 自动扩缩容
   - 服务网格（Istio）

2. **多智能体系统**：
   - 更复杂的智能体架构
   - 智能体通信协议
   - 任务调度和优先级
   - 智能体协作策略

3. **性能优化**：
   - 数据库缓存
   - CDN 集成
   - 请求队列
   - 批处理优化

4. **监控和运维**：
   - 日志聚合
   - 性能监控
   - 错误追踪
   - 成本分析

5. **安全性**：
   - API 身份验证
   - 请求限流
   - 输入验证
   - 数据加密

