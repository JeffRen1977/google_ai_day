# AI 智能体五日学习计划 (AI Agent 5-Day Learning Course)

基于 Google Gemini 的 AI 智能体（Agents）及其架构完整学习计划

## 📚 课程概述

本课程通过五天的时间，从基础概念到生产部署，全面学习 AI 智能体的开发与实践。课程使用 **Google Gemini** 模型作为核心 LLM，通过实践项目逐步掌握智能体的设计、开发、评估和部署。

### 课程特点

- ✅ **渐进式学习**: 从基础概念到生产部署，循序渐进
- ✅ **实践导向**: 每个概念都有对应的代码示例和实践项目
- ✅ **完整体系**: 涵盖智能体开发的完整生命周期
- ✅ **生产就绪**: 包含生产环境部署和优化最佳实践

### 技术栈

- **LLM**: Google Gemini (gemini-1.5-flash, gemini-1.5-pro)
- **语言**: Python 3.7+
- **框架**: FastAPI, Flask
- **工具**: Function Calling, RAG, Multi-Agent Systems

---

## 📅 课程进度与内容

### Day 1: 智能体简介与智能体架构 (Agents & Agentic Architectures) ✅

**学习目标**: 探索 AI 智能体的基础概念、定义特征，以及智能体架构与传统 LLM 应用的区别。

**核心内容**:
- ✅ 智能体的核心概念：感知-思考-行动（Perceive-Think-Act）循环
- ✅ 智能体架构：ReAct（Reasoning and Acting）模式
- ✅ 传统 LLM vs. 智能体：对比简单问答应用和具备决策能力的智能体

**实现文件**:
- `day1_agent_intro/simple_llm.py` - 传统 LLM 应用示例
- `day1_agent_intro/react_agent.py` - ReAct 智能体实现
- `day1_agent_intro/comparison_demo.py` - 对比演示程序
- `day1_agent_intro/model_utils.py` - 模型工具函数

**关键概念**:
- ReAct 循环：Reasoning → Action → Observation
- 思维链（Chain-of-Thought, CoT）
- 工具调用决策

**快速开始**:
```bash
cd day1_agent_intro
pip install -r requirements.txt
python react_agent.py
```

---

### Day 2: 智能体工具与 MCP 的互操作性 (Agent Tools & MCP Interoperability) ✅

**学习目标**: 理解 AI 智能体如何通过利用外部功能和 API 来"采取行动"，并探索工具发现和使用。

**核心内容**:
- ✅ 函数调用（Function Calling）：使用 Gemini API 的 `tools` 参数
- ✅ 工具互操作性：模拟 Model Context Protocol (MCP) 概念
- ✅ 多工具智能体：构建能够选择和使用多个工具的智能体

**实现文件**:
- `day2_agent_tools/function_calling_basic.py` - 基础函数调用示例
- `day2_agent_tools/multiple_tools_agent.py` - 多工具智能体
- `day2_agent_tools/weather_calculator_calendar.py` - 综合工具示例（天气、计算器、日历）

**关键概念**:
- Function Calling API
- 工具发现和选择
- 工具参数解析和执行

**快速开始**:
```bash
cd day2_agent_tools
pip install -r requirements.txt
python weather_calculator_calendar.py
```

---

### Day 3: 上下文工程：会话与内存管理 (Context Engineering: Sessions, Memory Management) ✅

**学习目标**: 探索如何构建可以记住过去交互并维持上下文的 AI 智能体。

**核心内容**:
- ✅ 短期记忆（会话历史）：使用 Gemini API 的聊天会话功能
- ✅ 长期记忆：检索增强生成 (RAG) 实现
- ✅ 上下文管理：总结旧聊天记录以节省上下文窗口

**实现文件**:
- `day3_context_memory/chat_sessions.py` - 聊天会话示例（短期记忆）
- `day3_context_memory/rag_memory.py` - RAG 长期记忆实现
- `day3_context_memory/context_summarization.py` - 上下文总结示例
- `day3_context_memory/combined_memory.py` - 结合短期和长期记忆

**关键概念**:
- Chat Sessions（聊天会话）
- RAG（Retrieval-Augmented Generation）
- 向量数据库和嵌入
- 上下文窗口优化

**快速开始**:
```bash
cd day3_context_memory
pip install -r requirements.txt
python combined_memory.py
```

---

### Day 4: 智能体质量：可观测性、日志、跟踪、评估与指标 (Agent Quality: Observability, Logging, Tracing, Evaluation, Metrics) ✅

**学习目标**: 掌握评估和改进智能体的关键学科，包括可观测性、日志、跟踪和评估策略。

**核心内容**:
- ✅ 日志与跟踪：记录智能体的每一步骤（感知、思考、工具调用、观察、行动）
- ✅ 评估指标：成功率、延迟、准确性、工具调用有效性
- ✅ Gemini 作为评估者：使用 Gemini 模型评估智能体输出

**实现文件**:
- `day4_agent_quality/logging_tracing.py` - 日志与跟踪示例
- `day4_agent_quality/evaluation_metrics.py` - 评估指标计算
- `day4_agent_quality/gemini_evaluator.py` - Gemini 评估者实现
- `day4_agent_quality/comprehensive_evaluation.py` - 综合评估系统

**关键概念**:
- 结构化日志记录
- 跟踪和可观测性
- 黄金数据集（Golden Dataset）
- 多维度评估（相关性、完整性、准确性、有用性）

**快速开始**:
```bash
cd day4_agent_quality
pip install -r requirements.txt
python comprehensive_evaluation.py
```

---

### Day 5: 从原型到生产 (Prototype to Production) ✅

**学习目标**: 学习部署和扩展 AI 智能体的最佳实践，包括多智能体系统和成本优化。

**核心内容**:
- ✅ API 部署：将智能体部署为 API 端点（FastAPI/Flask）
- ✅ 多智能体系统：Agent2Agent (A2A) 协议实现
- ✅ 扩展性与成本优化：缓存、异步处理、模型选择策略

**实现文件**:
- `day5_production/api_deployment.py` - API 部署示例（FastAPI）
- `day5_production/multi_agent_system.py` - 多智能体系统（Planner + Executor）
- `day5_production/scalability_optimization.py` - 扩展性与成本优化
- `day5_production/comprehensive_production.py` - 综合生产系统

**关键概念**:
- RESTful API 设计
- 多智能体系统架构（MAS）
- 缓存策略（TTL, LRU）
- 异步处理和批量处理
- 智能模型选择（Flash vs Pro）

**快速开始**:
```bash
cd day5_production
pip install -r requirements.txt
python api_deployment.py
# 访问 http://localhost:8000/docs
```

---

## 🚀 快速开始

### 环境设置

1. **克隆仓库**:
```bash
git clone <repository-url>
cd google_AI_day
```

2. **配置 API Key**:
```bash
# 在项目根目录创建 .env 文件
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

3. **安装依赖**:
```bash
# 安装所有天的依赖（或分别安装）
cd day1_agent_intro && pip install -r requirements.txt
cd ../day2_agent_tools && pip install -r requirements.txt
cd ../day3_context_memory && pip install -r requirements.txt
cd ../day4_agent_quality && pip install -r requirements.txt
cd ../day5_production && pip install -r requirements.txt
```

### 运行示例

每个 day 目录都有独立的示例程序，可以直接运行：

```bash
# Day 1: ReAct 智能体
python day1_agent_intro/react_agent.py

# Day 2: 多工具智能体
python day2_agent_tools/weather_calculator_calendar.py

# Day 3: 结合记忆系统
python day3_context_memory/combined_memory.py

# Day 4: 综合评估
python day4_agent_quality/comprehensive_evaluation.py

# Day 5: 生产 API
python day5_production/comprehensive_production.py
```

---

## 📖 学习路径

### 路径 1: 基础学习路径（推荐初学者）

1. **Day 1** → 理解智能体基础概念
2. **Day 2** → 学习工具调用
3. **Day 3** → 掌握内存管理
4. **Day 4** → 学习评估和监控
5. **Day 5** → 部署到生产环境

### 路径 2: 快速实践路径（有 LLM 基础）

1. **Day 1-2** → 快速了解基础和工具调用
2. **Day 3** → 深入内存管理
3. **Day 4-5** → 专注于生产部署和优化

### 路径 3: 深度研究路径（全面掌握）

1. 按顺序完成所有 5 天的内容
2. 深入研究每个 day 的 README 文档
3. 修改和扩展示例代码
4. 尝试实现自己的智能体项目

---

## 🎯 学习成果

完成本课程后，您将能够：

- ✅ 理解 AI 智能体的核心概念和架构模式
- ✅ 使用 Google Gemini API 构建智能体应用
- ✅ 实现工具调用和函数调用功能
- ✅ 构建具备短期和长期记忆的智能体
- ✅ 评估和监控智能体的性能
- ✅ 部署智能体到生产环境
- ✅ 优化智能体的成本和性能
- ✅ 设计多智能体系统

---

## 📁 项目结构

```
google_AI_day/
├── README.md                    # 本文件：课程总览
├── design_document             # 设计文档
├── .env                        # 环境变量（需要创建）
│
├── day1_agent_intro/           # Day 1: 智能体简介与架构
│   ├── simple_llm.py
│   ├── react_agent.py
│   ├── comparison_demo.py
│   ├── model_utils.py
│   ├── requirements.txt
│   └── README.md
│
├── day2_agent_tools/           # Day 2: 工具与函数调用
│   ├── function_calling_basic.py
│   ├── multiple_tools_agent.py
│   ├── weather_calculator_calendar.py
│   ├── model_utils.py
│   ├── requirements.txt
│   └── README.md
│
├── day3_context_memory/        # Day 3: 上下文与内存管理
│   ├── chat_sessions.py
│   ├── rag_memory.py
│   ├── context_summarization.py
│   ├── combined_memory.py
│   ├── model_utils.py
│   ├── requirements.txt
│   └── README.md
│
├── day4_agent_quality/         # Day 4: 质量、评估与监控
│   ├── logging_tracing.py
│   ├── evaluation_metrics.py
│   ├── gemini_evaluator.py
│   ├── comprehensive_evaluation.py
│   ├── model_utils.py
│   ├── requirements.txt
│   └── README.md
│
└── day5_production/            # Day 5: 生产部署与优化
    ├── api_deployment.py
    ├── multi_agent_system.py
    ├── scalability_optimization.py
    ├── comprehensive_production.py
    ├── model_utils.py
    ├── requirements.txt
    └── README.md
```

---

## 🔧 技术栈详情

### 核心依赖

- **google-generativeai**: Google Gemini API 客户端
- **python-dotenv**: 环境变量管理
- **fastapi**: 现代 Web API 框架
- **flask**: 传统 Web 框架（可选）
- **uvicorn**: ASGI 服务器
- **cachetools**: 缓存工具
- **aiohttp**: 异步 HTTP 客户端

### Gemini 模型

- **gemini-1.5-flash**: 快速、经济的模型（推荐用于简单任务）
- **gemini-1.5-pro**: 更强大的模型（推荐用于复杂任务）

---

## 📚 参考资料

### 官方文档

- [Google Gemini API 文档](https://ai.google.dev/docs)
- [Gemini API Python 客户端](https://github.com/google/generative-ai-python)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)

### 相关概念

- **ReAct**: Reasoning and Acting in Language Models
- **RAG**: Retrieval-Augmented Generation
- **Function Calling**: Tool use in LLMs
- **MCP**: Model Context Protocol (概念)
- **A2A**: Agent-to-Agent Protocol

---

## 🐛 故障排除

### 常见问题

1. **API Key 错误**
   - 确保 `.env` 文件在项目根目录
   - 检查 API Key 是否正确设置

2. **模型不可用**
   - 运行 `python model_utils.py` 查看可用模型
   - 检查 API Key 是否有权限访问相应模型

3. **依赖安装问题**
   - 使用 Python 3.7 或更高版本
   - 建议使用虚拟环境：`python -m venv venv && source venv/bin/activate`

4. **导入错误**
   - 确保已安装所有依赖：`pip install -r requirements.txt`
   - 检查是否在正确的目录运行脚本

---

## 📝 开发日志

### 完成状态

- ✅ **Day 1**: 智能体简介与架构 - 已完成
- ✅ **Day 2**: 工具与函数调用 - 已完成
- ✅ **Day 3**: 上下文与内存管理 - 已完成
- ✅ **Day 4**: 质量、评估与监控 - 已完成
- ✅ **Day 5**: 生产部署与优化 - 已完成

### 更新记录

- **2024-01**: 完成所有 5 天的课程内容
- 所有示例代码已测试通过
- 文档完整，包含详细的使用说明

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进本课程！

---

## 📄 许可证

本项目仅用于学习和教育目的。

---

## 🙏 致谢

- Google Gemini 团队提供的优秀 API
- 所有贡献者和学习者

---

**祝学习愉快！Happy Learning! 🚀**

