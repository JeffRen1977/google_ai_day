"""
Day 2 - 示例 1: 基础函数调用（Function Calling）
演示如何使用 Gemini API 的 tools 参数定义和调用外部工具。
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


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    模拟天气查询工具
    
    Args:
        location: 城市名称
        unit: 温度单位 ("celsius" 或 "fahrenheit")
        
    Returns:
        天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temp": 15, "condition": "晴朗", "humidity": 45},
        "上海": {"temp": 18, "condition": "多云", "humidity": 60},
        "广州": {"temp": 25, "condition": "小雨", "humidity": 75},
        "深圳": {"temp": 26, "condition": "晴朗", "humidity": 65},
    }
    
    location_key = location.replace("市", "").replace("省", "")
    
    if location_key in weather_data:
        data = weather_data[location_key]
        temp = data["temp"]
        if unit == "fahrenheit":
            temp = temp * 9/5 + 32
            unit_symbol = "°F"
        else:
            unit_symbol = "°C"
        
        return f"{location}的天气：温度 {temp}{unit_symbol}，{data['condition']}，湿度 {data['humidity']}%"
    else:
        return f"抱歉，未找到 {location} 的天气信息（模拟数据）"


def demonstrate_function_calling():
    """演示基础函数调用"""
    print("=" * 60)
    print("基础函数调用示例 - 天气查询工具")
    print("=" * 60)
    print()
    
    # 获取模型名称
    model_name = get_default_model()
    
    # 定义工具（函数声明）
    # 首先尝试使用 protos API（推荐方式）
    weather_tool = None
    use_protos = False
    
    try:
        import google.generativeai.protos as protos
        weather_tool = protos.Tool(
            function_declarations=[
                protos.FunctionDeclaration(
                    name="get_weather",
                    description="获取指定城市的天气信息，包括温度、天气状况和湿度",
                    parameters=protos.Schema(
                        type=protos.Type.OBJECT,
                        properties={
                            "location": protos.Schema(
                                type=protos.Type.STRING,
                                description="城市名称，例如：北京、上海、广州"
                            ),
                            "unit": protos.Schema(
                                type=protos.Type.STRING,
                                description="温度单位，'celsius'（摄氏度）或 'fahrenheit'（华氏度）",
                                enum=["celsius", "fahrenheit"]
                            )
                        },
                        required=["location"]
                    )
                )
            ]
        )
        use_protos = True
    except (AttributeError, ImportError):
        # protos API 不可用，使用字典格式（使用大写的类型名称）
        weather_tool = {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息，包括温度、天气状况和湿度",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "location": {
                                "type": "STRING",
                                "description": "城市名称，例如：北京、上海、广州"
                            },
                            "unit": {
                                "type": "STRING",
                                "description": "温度单位，'celsius'（摄氏度）或 'fahrenheit'（华氏度）",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            ]
        }
    
    # 创建模型并配置工具（在构造函数中传递）
    model = genai.GenerativeModel(model_name, tools=[weather_tool])
    
    # 用户查询
    user_query = "北京今天天气怎么样？"
    print(f"用户查询: {user_query}")
    print("-" * 60)
    
    # 使用 chat API 来管理对话和工具
    print("正在调用模型...", end="", flush=True)
    try:
        # 使用 chat API（工具已在模型中配置）
        chat = model.start_chat()
        response = chat.send_message(user_query)
        print(" 完成！\n")
        
        # 检查是否有函数调用请求
        function_calls = []
        response_text = ""
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # 检查是否是函数调用
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
                # 检查是否是文本响应
                elif hasattr(part, 'text') and part.text:
                    response_text += part.text
        
        if function_calls:
            print("检测到函数调用请求:")
            print("-" * 60)
            
            for func_call in function_calls:
                func_name = func_call.name
                # 将函数参数转换为字典
                args_dict = {}
                if hasattr(func_call, 'args'):
                    # 根据 API 版本，args 可能是不同的格式
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
                
                print(f"函数名称: {func_name}")
                print(f"参数: {json.dumps(args_dict, ensure_ascii=False, indent=2)}")
                print()
                
                # 执行函数调用
                if func_name == "get_weather":
                    location = args_dict.get("location", "")
                    unit = args_dict.get("unit", "celsius")
                    
                    print(f"执行函数: {func_name}(location='{location}', unit='{unit}')")
                    result = get_weather(location, unit)
                    print(f"函数返回: {result}")
                    print()
                    
                    # 将函数结果返回给模型
                    print("将结果返回给模型，生成最终答案...")
                    print("-" * 60)
                    
                    # 创建函数响应并使用 chat API 发送
                    try:
                        import google.generativeai.protos as protos
                        function_response = protos.FunctionResponse(
                            name=func_name,
                            response={"result": result}
                        )
                        # 使用 chat 发送函数响应
                        final_response = chat.send_message(function_response)
                    except (AttributeError, ImportError):
                        # protos API 不可用，使用字典格式
                        function_response = {
                            "function_response": {
                                "name": func_name,
                                "response": {"result": result}
                            }
                        }
                        try:
                            final_response = chat.send_message(function_response)
                        except:
                            # 如果都不支持，使用简单的文本拼接方式
                            final_response = chat.send_message(
                                f"函数调用结果: {result}\n\n请基于以上信息回答用户的问题。"
                            )
                    
                    # 提取最终答案
                    final_text = ""
                    if final_response.candidates and final_response.candidates[0].content.parts:
                        for part in final_response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                final_text += part.text
                    
                    if final_text:
                        print("最终答案:")
                        print(final_text)
                    else:
                        # 尝试其他方式获取响应
                        try:
                            print(final_response.text if hasattr(final_response, 'text') else str(final_response))
                        except:
                            print(str(final_response))
                else:
                    print(f"未知函数: {func_name}")
        else:
            # 没有函数调用，直接显示响应
            if response_text:
                print("模型直接回答（未调用函数）:")
                print(response_text)
            else:
                try:
                    print("模型响应:")
                    print(response.text if hasattr(response, 'text') else str(response))
                except:
                    print("无法解析响应，原始响应:")
                    print(str(response))
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_function_calling()

