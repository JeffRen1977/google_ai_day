# Day 4: 智能体质量：可观测性、日志、跟踪、评估与指标 (Agent Quality: Observability, Logging, Tracing, Evaluation, Metrics)

## 学习目标
- 掌握如何记录和跟踪智能体的每一步骤（感知、思考、工具调用、观察、行动）
- 学习评估智能体性能的关键指标（成功率、延迟、准确性、工具调用有效性）
- 了解如何使用 Gemini 模型本身作为评估者来评估智能体输出

## 文件说明
- `logging_tracing.py`: 日志与跟踪示例 - 记录智能体的每一步骤
- `evaluation_metrics.py`: 评估指标示例 - 计算成功率、延迟等指标
- `gemini_evaluator.py`: Gemini评估者示例 - 使用Gemini评估智能体输出
- `comprehensive_evaluation.py`: 综合示例 - 完整的评估系统

## 环境设置

1. 安装依赖：
```bash
cd day4_agent_quality
pip install -r requirements.txt
```

2. 配置 API Key：
```bash
# 在项目根目录（google_AI_day）创建 .env 文件（如果还没有）
# 填入你的 GEMINI_API_KEY:
# GEMINI_API_KEY=your_api_key_here
```

3. 运行示例：

**方式一：从 day4_agent_quality 目录运行**
```bash
cd day4_agent_quality
python3 logging_tracing.py          # 日志与跟踪示例
python3 evaluation_metrics.py      # 评估指标示例
python3 gemini_evaluator.py         # Gemini评估者示例
python3 comprehensive_evaluation.py # 综合评估示例
```

**方式二：从项目根目录运行**
```bash
# 从项目根目录（google_AI_day）
python3 day4_agent_quality/logging_tracing.py
python3 day4_agent_quality/evaluation_metrics.py
python3 day4_agent_quality/gemini_evaluator.py
python3 day4_agent_quality/comprehensive_evaluation.py
```

**注意**: 
- `.env` 文件应位于项目根目录（`google_AI_day/.env`），代码会自动从父目录加载环境变量。
- 代码会自动检测可用的 Gemini 模型（优先使用 `gemini-1.5-flash` 或 `gemini-1.5-pro`）。
- 在 macOS 上，如果 `python` 命令不可用，请使用 `python3`。

## 核心概念

### 1. 日志与跟踪（Logging & Tracing）

**概念**：
- 记录智能体的每个步骤：感知、思考、工具调用、观察、行动
- 捕获推理过程（Chain-of-Thought, CoT）
- 记录工具调用参数和结果
- 跟踪延迟和性能指标

**实现**：
```python
logger = AgentLogger()
logger.start_trace(trace_id, user_query)
logger.log_perception("收到用户查询")
logger.log_thinking("分析查询，确定工具")
logger.log_tool_call("calculate", {"expression": "25*4"})
logger.log_observation("工具返回结果: 100")
logger.log_response(answer, latency_ms)
logger.end_trace()
```

**跟踪步骤类型**：
- **感知 (Perception)**: 接收和理解输入
- **思考 (Thinking)**: 推理和决策过程
- **工具调用 (Tool Call)**: 调用外部工具
- **观察 (Observation)**: 工具执行结果
- **行动 (Action)**: 执行的操作
- **响应 (Response)**: 最终输出

### 2. 评估指标（Evaluation Metrics）

**关键指标**：

1. **成功率 (Success Rate)**
   - 定义：成功完成的测试用例百分比
   - 计算：`成功数 / 总测试数 × 100%`

2. **延迟 (Latency)**
   - 平均延迟：所有请求的平均响应时间
   - 最小/最大延迟：性能边界
   - 单位：毫秒 (ms)

3. **准确性 (Accuracy)**
   - 答案是否包含期望的关键信息
   - 答案类型是否正确（数字、文本、列表等）

4. **工具调用有效性 (Tool Call Effectiveness)**
   - 工具调用成功的百分比
   - 工具调用的正确性

**测试用例（黄金数据集）**：
```python
TestCase(
    query="请帮我计算 25 × 4 + 100 ÷ 5",
    expected_tool="calculate",
    expected_answer_contains=["120"],
    expected_answer_type="number"
)
```

### 3. Gemini 作为评估者

**概念**：
- 使用独立的 Gemini 实例评估智能体的输出
- 根据预定义标准评分（相关性、完整性、准确性、有用性）
- 提供详细的反馈和改进建议

**评估标准**：
- **相关性 (Relevance)**: 答案是否与问题相关
- **完整性 (Completeness)**: 答案是否完整回答了问题
- **准确性 (Accuracy)**: 答案是否准确
- **有用性 (Helpfulness)**: 答案是否对用户有帮助

**实现**：
```python
evaluator = GeminiEvaluator()
result = evaluator.evaluate(
    query="用户问题",
    expected_answer="期望答案（可选）",
    actual_answer="智能体的回答"
)
# result.score: 0-10分
# result.reasoning: 评估理由
# result.feedback: 改进建议
```

## 技术实现细节

### 日志记录器 (AgentLogger)

**功能**：
- 跟踪整个对话流程
- 记录每个步骤的时间戳
- 支持导出日志到文件
- 生成跟踪摘要

**日志格式**：
```json
{
  "trace_id": "trace_1234567890",
  "timestamp": "2024-01-15T10:30:00",
  "event": "trace_start",
  "user_query": "用户查询",
  "steps": [
    {
      "step_type": "perception",
      "timestamp": "2024-01-15T10:30:01",
      "data": {"observation": "..."}
    },
    ...
  ]
}
```

### 评估器 (AgentEvaluator)

**功能**：
- 执行测试用例
- 计算性能指标
- 生成评估报告
- 导出评估结果

**评估流程**：
1. 运行测试用例
2. 记录延迟和结果
3. 检查成功条件
4. 计算总体指标
5. 生成报告

### Gemini 评估者 (GeminiEvaluator)

**功能**：
- 使用 Gemini 模型评估回答质量
- 多维度评分（相关性、完整性、准确性、有用性）
- 提供详细反馈
- 支持批量评估

**评估提示结构**：
- 用户查询
- 智能体的回答
- 期望答案（可选）
- 上下文信息（可选）
- 评估标准
- JSON 格式输出要求

## 最佳实践

### 1. 日志记录
- ✅ 记录所有关键步骤
- ✅ 包含时间戳和跟踪ID
- ✅ 捕获推理过程（CoT）
- ✅ 记录工具调用参数
- ✅ 定期导出和归档日志

### 2. 评估指标
- ✅ 建立黄金数据集
- ✅ 定期运行评估
- ✅ 跟踪指标趋势
- ✅ 设置性能阈值
- ✅ 对比不同版本的性能

### 3. Gemini 评估
- ✅ 定义清晰的评估标准
- ✅ 使用一致的评估提示
- ✅ 结合多个评估维度
- ✅ 分析评估结果的一致性
- ✅ 根据反馈改进智能体

### 4. 综合评估系统
- ✅ 结合日志、指标和Gemini评估
- ✅ 生成综合报告
- ✅ 持续监控和改进
- ✅ 建立评估基准

## 测试状态

所有程序已通过测试，可以正常运行：

- ✅ `logging_tracing.py` - 日志与跟踪示例
- ✅ `evaluation_metrics.py` - 评估指标示例
- ✅ `gemini_evaluator.py` - Gemini评估者示例
- ✅ `comprehensive_evaluation.py` - 综合评估示例

### 已知问题

- 日志文件会保存在当前目录
- Gemini评估需要额外的API调用，可能增加成本
- 评估结果可能因模型版本而异

## 示例输出

### 日志跟踪示例输出：

```
[TRACE START] trace_1234567890: 请帮我计算 25 × 4 + 100 ÷ 5
[PERCEPTION] 收到用户查询: 请帮我计算 25 × 4 + 100 ÷ 5
[THINKING] 分析用户查询，确定需要使用的工具
[CoT] 查询内容: 请帮我计算 25 × 4 + 100 ÷ 5，需要分析是否需要工具调用
[TOOL CALL] calculate({"expression": "25 * 4 + 100 / 5"})
[OBSERVATION] 工具 calculate 返回: 计算结果: 25 * 4 + 100 / 5 = 120.0
[RESPONSE] 好的，25 × 4 + 100 ÷ 5 的计算结果是 120。 (延迟: 1234.56ms)
[TRACE END] trace_1234567890
```

### 评估指标示例输出：

```
============================================================
评估指标
============================================================
总测试数: 5
成功数: 4
失败数: 1
成功率: 80.00%

延迟指标:
  平均延迟: 1234.56ms
  最小延迟: 890.12ms
  最大延迟: 1567.89ms

工具调用:
  工具调用次数: 5
  工具调用有效性: 80.00%
```

### Gemini评估示例输出：

```
评估结果:
  总体评分: 8.5/10
  评估理由: 答案准确完整，直接回答了用户的问题...
  标准评分:
    - relevance: 9.0/10
    - completeness: 8.5/10
    - accuracy: 9.0/10
    - helpfulness: 8.0/10
  反馈: 建议可以添加更多上下文信息...
```

## 扩展学习

1. **高级日志系统**：
   - 结构化日志（JSON格式）
   - 日志聚合和分析
   - 实时监控和告警

2. **评估方法**：
   - A/B测试
   - 人类评估者
   - 自动化评估流水线

3. **性能优化**：
   - 延迟优化
   - 缓存策略
   - 批量处理

4. **监控和告警**：
   - 实时监控仪表板
   - 异常检测
   - 自动告警系统

