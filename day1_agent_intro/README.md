# Day 1: 智能体简介与智能体架构 (Agent Introduction & Architecture)

## 学习目标
- 探索 AI 智能体的基础概念和定义特征
- 理解智能体架构与传统 LLM 应用的区别
- 学习 ReAct (Reasoning and Acting) 架构模式

## 文件说明
- `simple_llm.py`: 传统 LLM 应用示例（仅生成文本）
- `react_agent.py`: ReAct 智能体示例（具备推理和行动能力）
- `comparison_demo.py`: 对比演示程序
- `model_utils.py`: 模型工具函数（自动检测可用的 Gemini 模型）

## 环境设置

1. 安装依赖：
```bash
cd day1_agent_intro
pip install -r requirements.txt
```

2. 配置 API Key：
```bash
# 在项目根目录（google_AI_day）创建 .env 文件
# 填入你的 GEMINI_API_KEY:
# GEMINI_API_KEY=your_api_key_here
```

3. 运行示例：

**方式一：从 day1_agent_intro 目录运行**
```bash
cd day1_agent_intro
python simple_llm.py          # 运行传统 LLM 示例
python react_agent.py         # 运行 ReAct 智能体示例
python comparison_demo.py     # 运行对比演示
```

**方式二：从项目根目录运行**
```bash
# 从项目根目录（google_AI_day）
python day1_agent_intro/simple_llm.py
python day1_agent_intro/react_agent.py
python day1_agent_intro/comparison_demo.py
```

**注意**: 
- `.env` 文件应位于项目根目录（`google_AI_day/.env`），代码会自动从父目录加载环境变量。
- 代码会自动检测可用的 Gemini 模型（优先使用 `gemini-1.5-flash` 或 `gemini-1.5-pro`）。
- 如果遇到模型相关错误，可以运行 `python model_utils.py` 查看可用模型列表。

## 核心概念

### 1. 智能体的核心特征
- **感知 (Perceive)**: 接收输入/观察环境
- **思考 (Think)**: 推理和决策
- **行动 (Act)**: 执行操作或调用工具

### 2. ReAct 模式
ReAct (Reasoning and Acting) 循环：
1. **Thought**: 推理当前情况
2. **Action**: 决定采取的行动
3. **Observation**: 观察行动结果
4. 重复直到完成任务

### 3. 传统 LLM vs 智能体
- **传统 LLM**: 仅生成文本响应，无状态，无外部工具调用
- **智能体**: 具备推理能力，可以调用工具，维护状态，多轮交互

