# Day 2: 智能体工具与 MCP 的互操作性 (Agent Tools & Interoperability with MCP)

## 学习目标
- 理解 AI 智能体如何通过利用外部功能和 API 来"采取行动"
- 学习函数调用（Function Calling）机制
- 探索工具发现和选择的过程
- 了解 Model Context Protocol (MCP) 的概念

## 文件说明
- `function_calling_basic.py`: 基础函数调用示例（单个工具）
- `multiple_tools_agent.py`: 多工具智能体（工具选择和调用）
- `weather_calculator_calendar.py`: 实践练习（天气、计算器、日历工具）

## 环境设置

1. 安装依赖：
```bash
cd day2_agent_tools
pip install -r requirements.txt
```

2. 配置 API Key：
```bash
# 在项目根目录（google_AI_day）创建 .env 文件（如果还没有）
# 填入你的 GEMINI_API_KEY:
# GEMINI_API_KEY=your_api_key_here
```

3. 运行示例：

**方式一：从 day2_agent_tools 目录运行**
```bash
cd day2_agent_tools
python function_calling_basic.py          # 基础函数调用示例
python multiple_tools_agent.py            # 多工具智能体示例
python weather_calculator_calendar.py     # 实践练习
```

**方式二：从项目根目录运行**
```bash
# 从项目根目录（google_AI_day）
python day2_agent_tools/function_calling_basic.py
python day2_agent_tools/multiple_tools_agent.py
python day2_agent_tools/weather_calculator_calendar.py
```

**注意**: 
- `.env` 文件应位于项目根目录（`google_AI_day/.env`），代码会自动从父目录加载环境变量。
- 代码会自动检测可用的 Gemini 模型（优先使用 `gemini-1.5-flash` 或 `gemini-1.5-pro`）。
- 在 macOS 上，如果 `python` 命令不可用，请使用 `python3`。

## 核心概念

### 1. 函数调用（Function Calling）
- 使用 Gemini API 的 `tools` 参数定义外部工具
- 模型可以识别何时需要调用函数
- 模型返回函数调用请求，由代码执行并返回结果

### 2. 工具互操作性
- 多个工具可以同时提供给模型
- 模型会根据用户请求智能选择需要调用的工具
- 模拟 Model Context Protocol (MCP) 的工具发现机制

### 3. 工具执行流程
1. **用户请求** → 模型分析
2. **工具选择** → 模型决定调用哪个工具
3. **函数调用** → 模型返回函数调用请求
4. **工具执行** → 代码执行函数
5. **结果返回** → 将结果返回给模型
6. **最终响应** → 模型生成最终答案

## 技术实现细节

### 工具定义格式

代码支持两种工具定义格式：

1. **Protobuf 格式**（优先使用，如果可用）：
   ```python
   import google.generativeai.protos as protos
   tool = protos.Tool(
       function_declarations=[
           protos.FunctionDeclaration(
               name="function_name",
               description="函数描述",
               parameters=protos.Schema(...)
           )
       ]
   )
   ```

2. **字典格式**（fallback，当 protos 不可用时）：
   ```python
   tool = {
       "function_declarations": [
           {
               "name": "function_name",
               "description": "函数描述",
               "parameters": {
                   "type": "OBJECT",      # 注意：使用大写
                   "properties": {
                       "param_name": {
                           "type": "STRING",  # 注意：使用大写
                           "description": "参数描述"
                       }
                   },
                   "required": ["param_name"]
               }
           }
       ]
   }
   ```

**重要提示**：
- 使用字典格式时，类型名称必须使用**大写**（`"OBJECT"`, `"STRING"` 等），而不是小写
- 工具应在模型构造函数中传递：`genai.GenerativeModel(model_name, tools=[tool])`
- 使用 `chat.start_chat()` 时，工具会自动从模型继承

### 函数响应处理

代码会根据 API 可用性自动选择响应格式：
- 优先使用 `protos.FunctionResponse`（如果可用）
- Fallback 到文本消息格式（当 protos 不可用时）

## 测试状态

所有程序已通过测试，可以正常运行：

- ✅ `function_calling_basic.py` - 基础函数调用示例
- ✅ `multiple_tools_agent.py` - 多工具智能体示例  
- ✅ `weather_calculator_calendar.py` - 实践练习

### 已知问题

- 在某些 Python 环境中，`google.generativeai.protos` 模块可能不可用，代码会自动使用字典格式作为 fallback
- 如果遇到 SSL 警告（NotOpenSSLWarning），不影响功能，可以忽略

## 示例输出

运行 `function_calling_basic.py` 的示例输出：

```
============================================================
基础函数调用示例 - 天气查询工具
============================================================

用户查询: 北京今天天气怎么样？
------------------------------------------------------------
正在调用模型... 完成！

检测到函数调用请求:
------------------------------------------------------------
函数名称: get_weather
参数: {
  "location": "北京",
  "unit": "celsius"
}

执行函数: get_weather(location='北京', unit='celsius')
函数返回: 北京的天气：温度 15°C，晴朗，湿度 45%

将结果返回给模型，生成最终答案...
------------------------------------------------------------
最终答案:
北京今天天气晴朗，温度为15摄氏度，湿度为45%。
```

