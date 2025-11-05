"""
Day 1 - 示例 1: 传统 LLM 应用
演示一个简单的问答应用，仅使用 LLM 生成文本响应，无工具调用能力。
"""

import os
import sys
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

def simple_llm_query(prompt: str, model_name: str = None, timeout: int = 60) -> str:
    """
    传统 LLM 应用：仅生成文本响应，无工具调用能力
    
    Args:
        prompt: 用户输入的问题
        model_name: 模型名称，如果为 None 则自动选择可用模型
        timeout: 超时时间（秒），默认 60 秒
        
    Returns:
        LLM 生成的文本响应
    """
    # 获取模型名称
    if model_name is None:
        model_name = get_default_model()
    
    print(f"正在调用模型 {model_name}...", end="", flush=True)
    
    # 创建模型实例
    model = genai.GenerativeModel(model_name)
    
    # 生成响应（单次调用，无状态管理）
    # 使用流式响应以便显示进度
    try:
        # 使用流式响应，可以显示进度
        response_stream = model.generate_content(prompt, stream=True)
        
        # 收集响应
        full_response = []
        start_time = time.time()
        chunk_count = 0
        
        for chunk in response_stream:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"请求超时（超过 {timeout} 秒）")
            
            # 处理 chunk 的文本内容
            chunk_text = None
            try:
                # 尝试直接访问 text 属性
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
            except (ValueError, AttributeError):
                # 如果 text 属性不可用，尝试从 parts 获取
                try:
                    if hasattr(chunk, 'parts') and chunk.parts:
                        for part in chunk.parts:
                            if hasattr(part, 'text'):
                                chunk_text = part.text
                                break
                except (AttributeError, IndexError):
                    pass
            
            # 如果还是无法获取，尝试从 candidates 获取
            if not chunk_text:
                try:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    chunk_text = part.text
                                    break
                except (AttributeError, IndexError):
                    pass
            
            if chunk_text:
                full_response.append(chunk_text)
                chunk_count += 1
                if chunk_count % 5 == 0:  # 每5个chunk显示一次进度
                    print(".", end="", flush=True)
        
        print(" 完成！")  # 响应完成
        
        # 合并所有响应
        response_text = "".join(full_response)
        if not response_text:
            raise ValueError("无法从流式响应中提取文本内容")
        return response_text
        
    except TimeoutError:
        print(f"\n错误: 请求超时（超过 {timeout} 秒）")
        raise
    except Exception as e:
        print(f"\n错误: {str(e)}")
        # 如果流式响应失败，尝试非流式响应
        print("尝试使用非流式响应...")
        try:
            response = model.generate_content(prompt)
            
            # 处理响应（兼容不同版本的 API）
            response_text = None
            
            # 方法1: 尝试直接访问 text（可能失败）
            try:
                if hasattr(response, 'text'):
                    response_text = response.text
            except (ValueError, AttributeError):
                pass
            
            # 方法2: 从 parts 获取
            if not response_text:
                try:
                    if hasattr(response, 'parts') and response.parts:
                        texts = [part.text for part in response.parts if hasattr(part, 'text')]
                        if texts:
                            response_text = "".join(texts)
                except (AttributeError, IndexError):
                    pass
            
            # 方法3: 从 candidates 获取
            if not response_text:
                try:
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            texts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                            if texts:
                                response_text = "".join(texts)
                except (AttributeError, IndexError):
                    pass
            
            if response_text:
                print(" 完成！")
                return response_text
            else:
                print(" 完成！")
                # 最后尝试转换为字符串
                return str(response)
        except Exception as e2:
            print(f"\n错误: 非流式响应也失败: {str(e2)}")
            raise

def demonstrate_agent_concepts():
    """演示智能体的核心概念"""
    print("=" * 60)
    print("传统 LLM 应用示例")
    print("=" * 60)
    print()
    
    # 让模型解释智能体的定义
    prompt1 = """
    请解释什么是 AI 智能体（Agent），并说明智能体的核心特征：
    1. 感知-思考-行动（Perceive-Think-Act）循环
    2. 与传统应用程序的根本区别
    
    请用中文回答。
    """
    
    print("问题 1: 什么是 AI 智能体？")
    print("-" * 60)
    response1 = simple_llm_query(prompt1)
    print(response1)
    print()
    
    # 简单的问答（无工具调用）
    prompt2 = "什么是 ReAct 模式？"
    print("问题 2: 什么是 ReAct 模式？")
    print("-" * 60)
    response2 = simple_llm_query(prompt2)
    print(response2)
    print()
    
    # 尝试让模型做需要外部工具的任务（但无法完成）
    prompt3 = "现在北京的时间是多少？"
    print("问题 3: 现在北京的时间是多少？")
    print("-" * 60)
    print("注意：传统 LLM 无法调用外部工具（如时间 API）")
    print("它只能基于训练数据回答，可能不准确：")
    response3 = simple_llm_query(prompt3)
    print(response3)
    print()
    
    print("=" * 60)
    print("传统 LLM 的限制：")
    print("- 无法调用外部工具或 API")
    print("- 无法获取实时信息")
    print("- 无状态管理（每次调用都是独立的）")
    print("- 无法执行实际操作（如计算、查询等）")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_agent_concepts()

