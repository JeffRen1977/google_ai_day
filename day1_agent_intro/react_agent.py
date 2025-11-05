"""
Day 1 - 示例 2: ReAct 智能体架构
演示一个基于 ReAct（Reasoning and Acting）模式的智能体：
- Thought: 推理当前情况
- Action: 决定采取的行动
- Observation: 观察行动结果
"""

import os
import re
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from model_utils import get_default_model

# 加载环境变量（从项目根目录）
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 配置 Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 GEMINI_API_KEY")

genai.configure(api_key=api_key)

# 模拟的工具函数
def get_current_time(location: str = "北京") -> str:
    """模拟获取当前时间的工具"""
    from datetime import datetime
    
    try:
        # 尝试使用 pytz（如果已安装）
        try:
            import pytz
            tz = pytz.timezone('Asia/Shanghai' if location == "北京" else 'UTC')
            current_time = datetime.now(tz)
            return f"{location}的当前时间是: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        except ImportError:
            # 如果没有 pytz，使用本地时间
            current_time = datetime.now()
            return f"{location}的当前时间是: {current_time.strftime('%Y-%m-%d %H:%M:%S')} (本地时间)"
    except Exception as e:
        return f"无法获取{location}的时间: {str(e)}"

def calculate(expression: str) -> str:
    """模拟计算器工具"""
    try:
        # 安全地评估数学表达式
        result = eval(expression.replace("×", "*").replace("÷", "/"))
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

def search_knowledge_base(query: str) -> str:
    """模拟知识库查询工具"""
    # 模拟的知识库
    knowledge_base = {
        "ReAct": "ReAct 是 Reasoning and Acting 的缩写，是一种智能体架构模式，结合了推理和行动能力。",
        "Agent": "AI 智能体是一个能够感知环境、进行推理并采取行动的自主系统。",
        "Gemini": "Gemini 是 Google 开发的大语言模型，支持多模态理解和函数调用。"
    }
    
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return f"知识库查询结果: {value}"
    return f"知识库中未找到关于 '{query}' 的信息"

# 可用工具映射
AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base
}

class ReActAgent:
    """ReAct 智能体实现"""
    
    def __init__(self, model_name: str = None):
        # 如果没有指定模型名称，使用默认可用模型
        if model_name is None:
            model_name = get_default_model()
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.max_iterations = 5  # 最大迭代次数
        self.history: List[Dict[str, str]] = []
    
    def parse_react_response(self, response_text: str) -> Tuple[str, str, str]:
        """
        解析 ReAct 格式的响应
        格式: Thought: ... Action: ... Observation: ...
        """
        thought = ""
        action = ""
        observation = ""
        
        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|$)', response_text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 提取 Action
        action_match = re.search(r'Action:\s*(.*?)(?=Observation:|$)', response_text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
        
        # 提取 Observation
        obs_match = re.search(r'Observation:\s*(.*?)(?=Thought:|$)', response_text, re.DOTALL)
        if obs_match:
            observation = obs_match.group(1).strip()
        
        return thought, action, observation
    
    def execute_action(self, action_text: str) -> str:
        """执行动作并返回观察结果"""
        action_text = action_text.strip()
        
        # 尝试解析工具调用
        # 格式: tool_name(arg1, arg2, ...)
        tool_match = re.match(r'(\w+)\s*\((.*?)\)', action_text)
        if tool_match:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2)
            
            if tool_name in AVAILABLE_TOOLS:
                # 解析参数
                args = [arg.strip().strip('"\'') for arg in args_str.split(',') if arg.strip()]
                
                # 调用工具
                try:
                    result = AVAILABLE_TOOLS[tool_name](*args)
                    return result
                except Exception as e:
                    return f"工具调用错误: {str(e)}"
            else:
                return f"未知工具: {tool_name}。可用工具: {', '.join(AVAILABLE_TOOLS.keys())}"
        
        return "无法解析动作格式。请使用格式: tool_name(arg1, arg2, ...)"
    
    def run(self, user_query: str) -> str:
        """
        运行 ReAct 循环处理用户查询
        
        Args:
            user_query: 用户的问题
            
        Returns:
            最终答案
        """
        print(f"\n用户查询: {user_query}")
        print("=" * 60)
        
        # 构建初始提示
        system_prompt = f"""你是一个基于 ReAct (Reasoning and Acting) 模式的 AI 智能体。

可用的工具：
1. get_current_time(location) - 获取指定地点的当前时间
2. calculate(expression) - 计算数学表达式，例如: calculate("2 + 3 * 4")
3. search_knowledge_base(query) - 在知识库中搜索信息

请按照以下格式思考和行动：

Thought: [你的推理过程，分析当前情况]
Action: [调用工具，格式: tool_name(arg1, arg2, ...)]
Observation: [观察工具执行的结果]

重复 Thought-Action-Observation 循环，直到你可以给出最终答案。

用户查询: {user_query}

开始思考和行动：
"""
        
        conversation_history = system_prompt
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- 迭代 {iteration} ---")
            
            # 调用模型生成响应
            print("正在生成响应...", end="", flush=True)
            start_time = time.time()
            timeout = 60  # 60秒超时
            
            try:
                # 使用流式响应以便显示进度
                response_stream = self.model.generate_content(conversation_history, stream=True)
                
                # 收集响应
                full_response = []
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
                        if chunk_count % 5 == 0:
                            print(".", end="", flush=True)
                
                response_text = "".join(full_response)
                if not response_text:
                    raise ValueError("无法从流式响应中提取文本内容")
                print(" 完成！")
                
            except (TimeoutError, Exception) as e:
                print(f"\n流式响应失败，尝试非流式响应: {str(e)}")
                # 回退到非流式响应
                try:
                    response = self.model.generate_content(conversation_history)
                    
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
                    
                    if not response_text:
                        response_text = str(response)
                    
                    print(" 完成！")
                except Exception as e2:
                    print(f"\n错误: 无法获取响应: {str(e2)}")
                    raise
            
            print(f"\n模型响应:\n{response_text}")
            
            # 解析响应
            thought, action, observation = self.parse_react_response(response_text)
            
            if thought:
                print(f"\n[Thought] {thought}")
            
            if action:
                print(f"[Action] {action}")
                # 执行动作
                obs_result = self.execute_action(action)
                print(f"[Observation] {obs_result}")
                
                # 更新对话历史
                conversation_history += f"\n\nThought: {thought}\nAction: {action}\nObservation: {obs_result}"
                
                # 检查是否完成（如果响应中包含 "Final Answer" 或类似标记）
                if "final answer" in response_text.lower() or "答案是" in response_text.lower():
                    break
            else:
                # 如果没有动作，可能是最终答案
                if "final answer" in response_text.lower() or "答案是" in response_text.lower():
                    break
                # 否则继续循环
                conversation_history += f"\n\n{response_text}"
        
        print("\n" + "=" * 60)
        print("ReAct 循环完成")
        print("=" * 60)
        
        return response_text

def demonstrate_react_agent():
    """演示 ReAct 智能体"""
    print("\n" + "=" * 60)
    print("ReAct 智能体示例")
    print("=" * 60)
    
    agent = ReActAgent()
    
    # 示例 1: 需要工具调用的任务
    print("\n示例 1: 查询时间（需要调用工具）")
    result1 = agent.run("现在北京的时间是多少？")
    
    # 示例 2: 需要计算的任务
    print("\n\n示例 2: 数学计算（需要调用工具）")
    agent2 = ReActAgent()
    result2 = agent2.run("请计算 25 × 4 + 100 ÷ 5 的结果")
    
    # 示例 3: 需要知识库查询的任务
    print("\n\n示例 3: 知识查询（需要调用工具）")
    agent3 = ReActAgent()
    result3 = agent3.run("什么是 ReAct 模式？")

if __name__ == "__main__":
    demonstrate_react_agent()

