"""
Day 2 - 示例 2: 多工具智能体（工具互操作性）
演示如何让模型在多个工具中选择并调用合适的工具。
模拟 Model Context Protocol (MCP) 的工具发现和选择过程。
"""

import os
import json
import time
from pathlib import Path
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


# 工具函数定义
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全地评估数学表达式
        result = eval(expression.replace("×", "*").replace("÷", "/").replace(" ", ""))
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def get_calendar_event(date: str = None) -> str:
    """获取日历事件"""
    # 模拟日历数据
    calendar_events = {
        "今天": ["团队会议 - 10:00", "代码审查 - 14:00"],
        "明天": ["项目演示 - 09:00", "客户会议 - 15:00"],
        "2024-01-15": ["重要会议 - 10:00"],
    }
    
    if date:
        date_key = date.replace(" ", "").replace("-", "")
        for key, events in calendar_events.items():
            if key in date or date in key:
                return f"{date}的日程: {', '.join(events)}"
    
    return "今天和明天的日程: " + json.dumps(calendar_events, ensure_ascii=False)


def get_weather(location: str) -> str:
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


class MultiToolAgent:
    """多工具智能体"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_default_model()
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        # 工具映射
        self.tools_map = {
            "calculate": calculate,
            "get_calendar_event": get_calendar_event,
            "get_weather": get_weather,
        }
        
        # 定义工具声明（使用字典格式，使用大写的类型名称）
        self.tools_declarations = [
            {
                "name": "calculate",
                "description": "计算数学表达式，支持加减乘除等基本运算",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "expression": {
                            "type": "STRING",
                            "description": "要计算的数学表达式，例如: '25 * 4 + 100 / 5'"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_calendar_event",
                "description": "查询指定日期的日历事件或日程安排",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "date": {
                            "type": "STRING",
                            "description": "日期，可以是'今天'、'明天'或具体日期如'2024-01-15'"
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
                            "description": "城市名称，例如：北京、上海、广州"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    
    def create_tool_proto(self):
        """创建工具 Protobuf 对象"""
        try:
            # 尝试使用 protos API
            function_declarations = []
            for tool_decl in self.tools_declarations:
                func_decl = genai.protos.FunctionDeclaration(
                    name=tool_decl["name"],
                    description=tool_decl["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            prop_name: genai.protos.Schema(
                                type=genai.protos.Type.STRING if prop["type"] == "string" else genai.protos.Type.NUMBER,
                                description=prop.get("description", "")
                            )
                            for prop_name, prop in tool_decl["parameters"]["properties"].items()
                        },
                        required=tool_decl["parameters"].get("required", [])
                    )
                )
                function_declarations.append(func_decl)
            
            return genai.protos.Tool(function_declarations=function_declarations)
        except Exception as e:
            print(f"警告: 无法创建 protobuf 工具对象: {e}")
            print("将使用简化版本...")
            return None
    
    def parse_function_call_args(self, func_call):
        """解析函数调用参数"""
        args_dict = {}
        if hasattr(func_call, 'args'):
            if hasattr(func_call.args, 'fields'):
                # Protobuf MapFields 格式
                for key, value in func_call.args.fields.items():
                    if hasattr(value, 'string_value'):
                        args_dict[key] = value.string_value
                    elif hasattr(value, 'number_value'):
                        args_dict[key] = value.number_value
                    elif hasattr(value, 'bool_value'):
                        args_dict[key] = value.bool_value
            elif isinstance(func_call.args, dict):
                args_dict = func_call.args
            else:
                try:
                    args_dict = dict(func_call.args)
                except:
                    pass
        return args_dict
    
    def execute_tool(self, func_name: str, args: dict) -> str:
        """执行工具函数"""
        if func_name in self.tools_map:
            func = self.tools_map[func_name]
            try:
                # 根据函数签名调用
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                # 构建参数
                call_args = {}
                for param in params:
                    if param in args:
                        call_args[param] = args[param]
                
                result = func(**call_args)
                return result
            except Exception as e:
                return f"工具执行错误: {str(e)}"
        else:
            return f"未知工具: {func_name}"
    
    def run(self, user_query: str) -> str:
        """运行智能体，处理用户查询"""
        print(f"\n用户查询: {user_query}")
        print("=" * 60)
        
        # 创建工具
        tools_proto = self.create_tool_proto()
        
        if tools_proto is None:
            # 如果无法创建 protobuf 对象，使用字典格式
            print("使用字典格式工具定义")
            print("可用工具:", ", ".join(self.tools_map.keys()))
            print()
            
            # 创建字典格式的工具定义
            tools_dict = {
                "function_declarations": self.tools_declarations
            }
            
            # 创建带工具的模型
            model_with_tools = genai.GenerativeModel(self.model_name, tools=[tools_dict])
            chat = model_with_tools.start_chat()
        else:
            # 使用 protobuf 工具对象
            chat = self.model.start_chat(tools=[tools_proto])
        
        # 使用 chat 进行对话
        print("正在分析请求并选择工具...", end="", flush=True)
        response = chat.send_message(user_query)
        print(" 完成！\n")
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"--- 迭代 {iteration} ---")
            
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
                print(f"检测到 {len(function_calls)} 个函数调用:")
                print("-" * 60)
                
                # 执行所有函数调用
                function_responses = []
                for func_call in function_calls:
                    func_name = func_call.name
                    args = self.parse_function_call_args(func_call)
                    
                    print(f"[工具调用] {func_name}({json.dumps(args, ensure_ascii=False)})")
                    result = self.execute_tool(func_name, args)
                    print(f"[工具返回] {result}")
                    print()
                    
                    # 创建函数响应
                    try:
                        import google.generativeai.protos as protos
                        function_responses.append(
                            protos.FunctionResponse(
                                name=func_name,
                                response={"result": result}
                            )
                        )
                    except (AttributeError, ImportError):
                        # protos不可用，使用文本消息作为fallback
                        function_responses.append(
                            f"函数 {func_name} 的返回结果: {result}"
                        )
                
                # 将函数响应发送回模型
                print("将工具结果返回给模型...")
                response = chat.send_message(function_responses)
            else:
                # 没有函数调用，返回最终答案
                if response_text:
                    print("最终答案:")
                    print(response_text)
                    return response_text
                else:
                    try:
                        text = response.text if hasattr(response, 'text') else str(response)
                        print("最终答案:")
                        print(text)
                        return text
                    except:
                        print("响应:")
                        print(str(response))
                        return str(response)
        
        print("达到最大迭代次数")
        return response_text if response_text else str(response)


def demonstrate_multi_tool_agent():
    """演示多工具智能体"""
    print("\n" + "=" * 60)
    print("多工具智能体示例 - 工具互操作性")
    print("=" * 60)
    
    agent = MultiToolAgent()
    
    # 示例 1: 需要计算器工具
    print("\n示例 1: 数学计算")
    print("-" * 60)
    result1 = agent.run("请帮我计算 25 × 4 + 100 ÷ 5 的结果")
    
    # 示例 2: 需要日历工具
    print("\n\n示例 2: 日历查询")
    print("-" * 60)
    agent2 = MultiToolAgent()
    result2 = agent2.run("我今天有什么日程安排？")
    
    # 示例 3: 需要天气工具
    print("\n\n示例 3: 天气查询")
    print("-" * 60)
    agent3 = MultiToolAgent()
    result3 = agent3.run("上海今天的天气怎么样？")
    
    # 示例 4: 可能需要多个工具
    print("\n\n示例 4: 综合查询（可能需要多个工具）")
    print("-" * 60)
    agent4 = MultiToolAgent()
    result4 = agent4.run("请计算 (100 + 50) × 2，然后告诉我明天的日程")


if __name__ == "__main__":
    demonstrate_multi_tool_agent()

