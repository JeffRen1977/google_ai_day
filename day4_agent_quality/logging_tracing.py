"""
Day 4 - 示例 1: 日志与跟踪
演示如何记录智能体的每一步骤（感知、思考、工具调用、观察、行动），
捕获模型的推理过程（CoT）和工具调用参数。
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from model_utils import get_default_model

# 加载环境变量（从项目根目录）
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 配置 Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 GEMINI_API_KEY")

genai.configure(api_key=api_key)


class AgentLogger:
    """智能体日志记录器"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.logs: List[Dict[str, Any]] = []
        self.current_trace_id = None
    
    def start_trace(self, trace_id: str, user_query: str):
        """开始新的跟踪"""
        self.current_trace_id = trace_id
        log_entry = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "event": "trace_start",
            "user_query": user_query,
            "steps": []
        }
        self.logs.append(log_entry)
        self._write_log(f"[TRACE START] {trace_id}: {user_query}")
    
    def log_perception(self, observation: str):
        """记录感知步骤"""
        self._log_step("perception", {"observation": observation})
        self._write_log(f"[PERCEPTION] {observation}")
    
    def log_thinking(self, reasoning: str, thought_chain: Optional[str] = None):
        """记录思考步骤（推理过程）"""
        self._log_step("thinking", {
            "reasoning": reasoning,
            "thought_chain": thought_chain
        })
        self._write_log(f"[THINKING] {reasoning}")
        if thought_chain:
            self._write_log(f"[CoT] {thought_chain}")
    
    def log_tool_call(self, tool_name: str, parameters: Dict[str, Any]):
        """记录工具调用"""
        self._log_step("tool_call", {
            "tool_name": tool_name,
            "parameters": parameters
        })
        self._write_log(f"[TOOL CALL] {tool_name}({json.dumps(parameters, ensure_ascii=False)})")
    
    def log_observation(self, observation: str):
        """记录观察结果"""
        self._log_step("observation", {"observation": observation})
        self._write_log(f"[OBSERVATION] {observation}")
    
    def log_action(self, action: str, result: Optional[str] = None):
        """记录行动"""
        self._log_step("action", {
            "action": action,
            "result": result
        })
        self._write_log(f"[ACTION] {action}")
        if result:
            self._write_log(f"[RESULT] {result}")
    
    def log_response(self, response: str, latency_ms: float):
        """记录最终响应"""
        self._log_step("response", {
            "response": response,
            "latency_ms": latency_ms
        })
        self._write_log(f"[RESPONSE] {response} (延迟: {latency_ms:.2f}ms)")
    
    def _log_step(self, step_type: str, data: Dict[str, Any]):
        """记录步骤"""
        if not self.logs:
            return
        
        step = {
            "step_type": step_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.logs[-1]["steps"].append(step)
    
    def _write_log(self, message: str):
        """写入日志文件"""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} {message}\n")
    
    def end_trace(self):
        """结束跟踪"""
        if self.logs:
            self.logs[-1]["event"] = "trace_end"
            self.logs[-1]["end_timestamp"] = datetime.now().isoformat()
            self._write_log(f"[TRACE END] {self.current_trace_id}")
            self.current_trace_id = None
    
    def get_trace_summary(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """获取跟踪摘要"""
        if trace_id:
            for log in self.logs:
                if log.get("trace_id") == trace_id:
                    return self._summarize_trace(log)
        elif self.logs:
            return self._summarize_trace(self.logs[-1])
        return {}
    
    def _summarize_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """总结跟踪信息"""
        steps = trace.get("steps", [])
        return {
            "trace_id": trace.get("trace_id"),
            "duration": self._calculate_duration(trace),
            "total_steps": len(steps),
            "step_types": [step["step_type"] for step in steps],
            "tool_calls": [
                step["data"] for step in steps 
                if step["step_type"] == "tool_call"
            ],
            "thinking_steps": [
                step["data"] for step in steps 
                if step["step_type"] == "thinking"
            ]
        }
    
    def _calculate_duration(self, trace: Dict[str, Any]) -> float:
        """计算跟踪持续时间（秒）"""
        start = datetime.fromisoformat(trace["timestamp"])
        end_str = trace.get("end_timestamp")
        if end_str:
            end = datetime.fromisoformat(end_str)
            return (end - start).total_seconds()
        return 0.0
    
    def export_logs(self, filename: str):
        """导出日志到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        print(f"日志已导出到: {filename}")


class LoggedMultiToolAgent:
    """带日志记录的多工具智能体"""
    
    def __init__(self, model_name: str = None, logger: Optional[AgentLogger] = None):
        if model_name is None:
            model_name = get_default_model()
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.logger = logger or AgentLogger()
        
        # 工具映射
        self.tools_map = {
            "calculate": self._calculate,
            "get_calendar_event": self._get_calendar_event,
            "get_weather": self._get_weather,
        }
        
        # 工具声明
        self.tools_declarations = [
            {
                "name": "calculate",
                "description": "计算数学表达式，支持加减乘除等基本运算",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "expression": {
                            "type": "STRING",
                            "description": "要计算的数学表达式"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_calendar_event",
                "description": "查询指定日期的日历事件",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "date": {
                            "type": "STRING",
                            "description": "日期"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "location": {
                            "type": "STRING",
                            "description": "城市名称"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    
    def _calculate(self, expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression.replace("×", "*").replace("÷", "/").replace(" ", ""))
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    def _get_calendar_event(self, date: str = None) -> str:
        """获取日历事件"""
        calendar_events = {
            "今天": ["团队会议 - 10:00", "代码审查 - 14:00"],
            "明天": ["项目演示 - 09:00", "客户会议 - 15:00"],
        }
        if date:
            for key, events in calendar_events.items():
                if key in date or date in key:
                    return f"{date}的日程: {', '.join(events)}"
        return "今天和明天的日程: " + json.dumps(calendar_events, ensure_ascii=False)
    
    def _get_weather(self, location: str) -> str:
        """获取天气信息"""
        weather_data = {
            "北京": "晴朗，15°C",
            "上海": "多云，18°C",
            "广州": "小雨，25°C",
        }
        location_clean = location.replace("市", "").replace("省", "")
        if location_clean in weather_data:
            return f"{location}的天气: {weather_data[location_clean]}"
        return f"{location}的天气: 未找到数据（模拟）"
    
    def run(self, user_query: str, trace_id: Optional[str] = None) -> str:
        """运行智能体，记录所有步骤"""
        start_time = time.time()
        
        # 生成跟踪ID
        if not trace_id:
            trace_id = f"trace_{int(time.time())}"
        
        # 开始跟踪
        self.logger.start_trace(trace_id, user_query)
        
        # 1. 感知：接收用户查询
        self.logger.log_perception(f"收到用户查询: {user_query}")
        
        # 2. 思考：分析查询
        self.logger.log_thinking(
            "分析用户查询，确定需要使用的工具",
            f"查询内容: {user_query}，需要分析是否需要工具调用"
        )
        
        # 创建工具
        tools_dict = {
            "function_declarations": self.tools_declarations
        }
        model_with_tools = genai.GenerativeModel(self.model_name, tools=[tools_dict])
        chat = model_with_tools.start_chat()
        
        try:
            # 发送消息
            response = chat.send_message(user_query)
            
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # 检查函数调用
                function_calls = []
                response_text = ""
                
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call)
                        elif hasattr(part, 'text') and part.text:
                            response_text += part.text
                
                if function_calls:
                    # 3. 工具调用
                    for func_call in function_calls:
                        func_name = func_call.name
                        
                        # 解析参数
                        args_dict = {}
                        if hasattr(func_call, 'args'):
                            if hasattr(func_call.args, 'fields'):
                                for key, value in func_call.args.fields.items():
                                    if hasattr(value, 'string_value'):
                                        args_dict[key] = value.string_value
                                    elif hasattr(value, 'number_value'):
                                        args_dict[key] = value.number_value
                            elif isinstance(func_call.args, dict):
                                args_dict = func_call.args
                        
                        # 记录工具调用
                        self.logger.log_tool_call(func_name, args_dict)
                        
                        # 执行工具
                        if func_name in self.tools_map:
                            tool_func = self.tools_map[func_name]
                            result = tool_func(**args_dict)
                            
                            # 4. 观察：记录工具执行结果
                            self.logger.log_observation(f"工具 {func_name} 返回: {result}")
                            
                            # 5. 行动：将结果发送回模型
                            try:
                                import google.generativeai.protos as protos
                                function_response = protos.FunctionResponse(
                                    name=func_name,
                                    response={"result": result}
                                )
                                response = chat.send_message(function_response)
                            except (AttributeError, ImportError):
                                response = chat.send_message(
                                    f"函数 {func_name} 的返回结果: {result}"
                                )
                else:
                    # 6. 最终响应
                    if response_text:
                        latency_ms = (time.time() - start_time) * 1000
                        self.logger.log_response(response_text, latency_ms)
                        self.logger.end_trace()
                        return response_text
                    else:
                        text = response.text if hasattr(response, 'text') else str(response)
                        latency_ms = (time.time() - start_time) * 1000
                        self.logger.log_response(text, latency_ms)
                        self.logger.end_trace()
                        return text
            
            self.logger.end_trace()
            return response_text if response_text else "处理完成"
            
        except Exception as e:
            self.logger.log_action("error", str(e))
            self.logger.end_trace()
            raise


def demonstrate_logging_tracing():
    """演示日志和跟踪功能"""
    print("=" * 60)
    print("日志与跟踪示例")
    print("=" * 60)
    print()
    
    # 创建日志记录器
    logger = AgentLogger(log_file="agent_logs.txt")
    
    # 创建带日志的智能体
    agent = LoggedMultiToolAgent(logger=logger)
    
    # 测试查询
    test_queries = [
        "请帮我计算 25 × 4 + 100 ÷ 5",
        "我今天有什么日程安排？",
        "北京今天的天气怎么样？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试查询 {i}: {query}")
        print(f"{'='*60}\n")
        
        try:
            result = agent.run(query, trace_id=f"test_{i}")
            print(f"\n结果: {result}")
            
            # 显示跟踪摘要
            summary = logger.get_trace_summary()
            print(f"\n跟踪摘要:")
            print(f"  - 总步骤数: {summary.get('total_steps', 0)}")
            print(f"  - 持续时间: {summary.get('duration', 0):.2f}秒")
            print(f"  - 工具调用: {len(summary.get('tool_calls', []))}次")
            print(f"  - 思考步骤: {len(summary.get('thinking_steps', []))}次")
            
        except Exception as e:
            print(f"错误: {e}")
    
    # 导出日志
    logger.export_logs("agent_logs_export.json")
    
    print("\n" + "=" * 60)
    print("日志分析：")
    print("- 所有步骤都被记录（感知、思考、工具调用、观察、行动）")
    print("- 推理过程（CoT）被捕获")
    print("- 工具调用参数被完整记录")
    print("- 延迟和性能指标被跟踪")
    print("- 日志已保存到文件")


if __name__ == "__main__":
    try:
        demonstrate_logging_tracing()
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

