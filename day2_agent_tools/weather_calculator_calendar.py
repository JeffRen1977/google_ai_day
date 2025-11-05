"""
Day 2 - 示例 3: 实践练习 - 天气、计算器、日历工具
构建一个智能体，能够根据用户请求选择调用计算器工具或日历查询工具。
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


# 工具函数
def calculator(expression: str) -> dict:
    """
    计算器工具 - 执行数学计算
    
    Args:
        expression: 数学表达式，例如 "25 * 4 + 100 / 5"
        
    Returns:
        包含计算结果的字典
    """
    try:
        # 清理表达式
        expr_clean = expression.replace("×", "*").replace("÷", "/").replace(" ", "")
        result = eval(expr_clean)
        return {
            "success": True,
            "expression": expression,
            "result": result,
            "message": f"{expression} = {result}"
        }
    except Exception as e:
        return {
            "success": False,
            "expression": expression,
            "error": str(e),
            "message": f"计算错误: {str(e)}"
        }


def calendar_query(date: str = None) -> dict:
    """
    日历查询工具 - 查询指定日期的日程安排
    
    Args:
        date: 日期，可以是 "今天"、"明天" 或具体日期
        
    Returns:
        包含日程信息的字典
    """
    # 模拟日历数据
    calendar_data = {
        "今天": {
            "date": "2024-01-15",
            "events": [
                {"time": "10:00", "title": "团队会议", "location": "会议室A"},
                {"time": "14:00", "title": "代码审查", "location": "在线"},
                {"time": "16:30", "title": "项目演示准备", "location": "办公室"}
            ]
        },
        "明天": {
            "date": "2024-01-16",
            "events": [
                {"time": "09:00", "title": "项目演示", "location": "会议室B"},
                {"time": "15:00", "title": "客户会议", "location": "客户公司"}
            ]
        },
        "2024-01-17": {
            "date": "2024-01-17",
            "events": [
                {"time": "11:00", "title": "技术分享", "location": "在线"}
            ]
        }
    }
    
    # 查找匹配的日期
    date_key = None
    if date:
        date_lower = date.lower().replace(" ", "")
        for key in calendar_data.keys():
            if key in date_lower or date_lower in key.lower():
                date_key = key
                break
    
    if not date_key:
        date_key = "今天"  # 默认返回今天的日程
    
    data = calendar_data[date_key]
    return {
        "success": True,
        "date": data["date"],
        "query": date or "今天",
        "events": data["events"],
        "message": f"{date_key}的日程: " + ", ".join([f"{e['time']} {e['title']}" for e in data["events"]])
    }


class ToolSelectorAgent:
    """工具选择智能体 - 能够根据用户请求选择合适的工具"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_default_model()
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        # 工具映射
        self.tools = {
            "calculator": calculator,
            "calendar_query": calendar_query,
        }
    
    def create_tools_proto(self):
        """创建工具 Protobuf 对象或字典格式"""
        try:
            function_declarations = [
                genai.protos.FunctionDeclaration(
                    name="calculator",
                    description="执行数学计算，支持加减乘除等基本运算。用于计算数学表达式。",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "expression": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="要计算的数学表达式，例如: '25 * 4 + 100 / 5' 或 '100 + 50' * 2"
                            )
                        },
                        required=["expression"]
                    )
                ),
                genai.protos.FunctionDeclaration(
                    name="calendar_query",
                    description="查询指定日期的日历事件或日程安排。用于查看日程。",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "date": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="日期，可以是'今天'、'明天'或具体日期如'2024-01-15'。如果不提供，默认查询今天的日程。"
                            )
                        },
                        required=[]
                    )
                )
            ]
            
            return genai.protos.Tool(function_declarations=function_declarations)
        except (AttributeError, ImportError):
            # protos不可用，使用字典格式
            return {
                "function_declarations": [
                    {
                        "name": "calculator",
                        "description": "执行数学计算，支持加减乘除等基本运算。用于计算数学表达式。",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "expression": {
                                    "type": "STRING",
                                    "description": "要计算的数学表达式，例如: '25 * 4 + 100 / 5' 或 '100 + 50' * 2"
                                }
                            },
                            "required": ["expression"]
                        }
                    },
                    {
                        "name": "calendar_query",
                        "description": "查询指定日期的日历事件或日程安排。用于查看日程。",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "date": {
                                    "type": "STRING",
                                    "description": "日期，可以是'今天'、'明天'或具体日期如'2024-01-15'。如果不提供，默认查询今天的日程。"
                                }
                            },
                            "required": []
                        }
                    }
                ]
            }
    
    def parse_args(self, func_call):
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
                # 字典格式
                args_dict = func_call.args
            else:
                # 尝试转换为字典
                try:
                    args_dict = dict(func_call.args)
                except:
                    pass
        return args_dict
    
    def process_query(self, user_query: str) -> str:
        """处理用户查询"""
        print(f"\n用户查询: {user_query}")
        print("=" * 60)
        
        tools_proto = self.create_tools_proto()
        if tools_proto is None:
            return "无法初始化工具系统"
        
        print("正在分析请求...", end="", flush=True)
        # 如果tools_proto是字典，需要在模型构造函数中传递
        if isinstance(tools_proto, dict):
            model_with_tools = genai.GenerativeModel(self.model_name, tools=[tools_proto])
            chat = model_with_tools.start_chat()
        else:
            chat = self.model.start_chat(tools=[tools_proto])
        response = chat.send_message(user_query)
        print(" 完成！\n")
        
        max_iterations = 3
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
                print(f"第 {iteration} 轮 - 检测到工具调用:")
                print("-" * 60)
                
                function_responses = []
                for func_call in function_calls:
                    func_name = func_call.name
                    args = self.parse_args(func_call)
                    
                    print(f"工具: {func_name}")
                    print(f"参数: {json.dumps(args, ensure_ascii=False)}")
                    
                    # 执行工具
                    if func_name in self.tools:
                        tool_func = self.tools[func_name]
                        
                        # 如果参数为空，检查函数签名并提示错误
                        if not args:
                            import inspect
                            sig = inspect.signature(tool_func)
                            params = list(sig.parameters.keys())
                            required_params = [p for p, param in sig.parameters.items() 
                                             if param.default == inspect.Parameter.empty]
                            if required_params:
                                print(f"错误: 缺少必需参数 {required_params}")
                                continue
                        
                        try:
                            result = tool_func(**args)
                        except TypeError as e:
                            print(f"错误: 函数调用失败 - {e}")
                            # 尝试使用函数签名获取默认值
                            import inspect
                            sig = inspect.signature(tool_func)
                            bound_args = {}
                            for param_name, param in sig.parameters.items():
                                if param_name in args:
                                    bound_args[param_name] = args[param_name]
                                elif param.default != inspect.Parameter.empty:
                                    bound_args[param_name] = param.default
                            try:
                                result = tool_func(**bound_args)
                            except Exception as e2:
                                print(f"错误: 即使使用默认参数也失败 - {e2}")
                                continue
                        print(f"结果: {result.get('message', str(result))}")
                        print()
                        
                        try:
                            import google.generativeai.protos as protos
                            function_responses.append(
                                protos.FunctionResponse(
                                    name=func_name,
                                    response=result
                                )
                            )
                        except (AttributeError, ImportError):
                            # protos不可用，使用文本消息作为fallback
                            function_responses.append(
                                f"函数 {func_name} 的返回结果: {result.get('message', str(result))}"
                            )
                    else:
                        print(f"错误: 未知工具 {func_name}")
                
                # 将结果发送回模型
                if function_responses:
                    print("将结果返回给模型...")
                    response = chat.send_message(function_responses)
            else:
                # 最终答案
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
                        return str(response)
        
        return response_text if response_text else "处理完成"


def demonstrate_practice():
    """演示实践练习"""
    print("\n" + "=" * 60)
    print("实践练习 - 工具选择智能体")
    print("=" * 60)
    print("\n可用工具:")
    print("1. calculator - 数学计算器")
    print("2. calendar_query - 日历查询")
    print()
    
    agent = ToolSelectorAgent()
    
    # 测试用例
    test_cases = [
        "请帮我计算 25 × 4 + 100 ÷ 5",
        "我今天有什么日程安排？",
        "请计算 (100 + 50) × 2",
        "明天的日程是什么？",
        "先计算 15 * 23，然后告诉我明天的日程",
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试用例 {i}: {query}")
        print(f"{'='*60}")
        result = agent.process_query(query)
        print()
    
    print("\n" + "=" * 60)
    print("实践练习完成")
    print("=" * 60)
    print("\n总结:")
    print("- 智能体能够根据用户请求自动选择合适的工具")
    print("- 支持单个工具调用和多个工具的组合调用")
    print("- 实现了工具发现和选择的过程（模拟 MCP 协议）")


if __name__ == "__main__":
    demonstrate_practice()

