"""
Day 3 - 示例 1: 短期记忆（会话历史）
演示如何使用 Gemini API 的聊天会话功能来维持对话上下文。
"""

import os
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


def demonstrate_chat_sessions():
    """演示聊天会话的短期记忆功能"""
    print("=" * 60)
    print("短期记忆示例 - 聊天会话（Chat Sessions）")
    print("=" * 60)
    print()
    
    # 获取模型
    model_name = get_default_model()
    model = genai.GenerativeModel(model_name)
    
    # 创建聊天会话
    print("创建新的聊天会话...")
    chat = model.start_chat()
    print("聊天会话已创建\n")
    
    # 多轮对话示例
    conversations = [
        "我的名字是张三，我是一名软件工程师。",
        "我喜欢编程，特别是Python和JavaScript。",
        "我最喜欢的编程语言是什么？",
        "我今年多少岁？",
        "请总结一下关于我的信息。"
    ]
    
    print("开始多轮对话：")
    print("-" * 60)
    
    for i, user_message in enumerate(conversations, 1):
        print(f"\n[轮次 {i}] 用户: {user_message}")
        print("-" * 60)
        
        try:
            # 发送消息到聊天会话
            response = chat.send_message(user_message)
            
            # 获取响应文本
            if response.candidates and response.candidates[0].content.parts:
                assistant_text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        assistant_text += part.text
                
                if assistant_text:
                    print(f"助手: {assistant_text}")
                else:
                    print(f"助手: {response.text if hasattr(response, 'text') else str(response)}")
            else:
                print(f"助手: {response.text if hasattr(response, 'text') else str(response)}")
            
            # 显示对话历史统计
            history = chat.history
            print(f"\n[对话历史统计] 消息数量: {len(history)}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("对话完成")
    print("=" * 60)
    print("\n分析：")
    print("- 模型能够记住之前对话中提到的信息（名字、职业、兴趣）")
    print("- 即使没有明确提到年龄，模型也能识别并指出未提供该信息")
    print("- 模型能够总结整个对话的内容")
    print("- 聊天会话自动维护了对话历史，无需手动管理")


def demonstrate_context_retention():
    """演示上下文保持能力"""
    print("\n" + "=" * 60)
    print("上下文保持示例")
    print("=" * 60)
    print()
    
    model_name = get_default_model()
    model = genai.GenerativeModel(model_name)
    chat = model.start_chat()
    
    # 设置场景
    print("场景：帮助用户规划旅行")
    print("-" * 60)
    
    scenario = [
        "我想去日本旅行，计划7天时间。",
        "我喜欢历史文化和美食。",
        "请推荐一些适合我的景点。",
        "我对东京的哪些区域特别感兴趣？",
        "请为我制定一个7天的行程安排，包括我提到的兴趣点。"
    ]
    
    for i, message in enumerate(scenario, 1):
        print(f"\n[轮次 {i}] 用户: {message}")
        try:
            response = chat.send_message(message)
            
            if response.candidates and response.candidates[0].content.parts:
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text += part.text
                
                if text:
                    print(f"助手: {text[:200]}..." if len(text) > 200 else f"助手: {text}")
            else:
                text = response.text if hasattr(response, 'text') else str(response)
                print(f"助手: {text[:200]}..." if len(text) > 200 else f"助手: {text}")
                
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n" + "=" * 60)
    print("分析：")
    print("- 模型能够记住用户提到的所有偏好（日本、7天、历史、文化、美食）")
    print("- 在后续对话中，模型能够引用之前提到的信息")
    print("- 模型能够整合多个回合的信息来提供综合性的回答")


def demonstrate_chat_history_access():
    """演示如何访问和管理聊天历史"""
    print("\n" + "=" * 60)
    print("聊天历史访问示例")
    print("=" * 60)
    print()
    
    model_name = get_default_model()
    model = genai.GenerativeModel(model_name)
    chat = model.start_chat()
    
    # 进行几轮对话
    messages = [
        "我叫李四",
        "我是一名数据科学家",
        "我工作的公司是Google"
    ]
    
    for msg in messages:
        chat.send_message(msg)
    
    # 访问聊天历史
    print("聊天历史内容：")
    print("-" * 60)
    
    history = chat.history
    for i, content in enumerate(history, 1):
        role = content.role if hasattr(content, 'role') else 'unknown'
        print(f"\n[{i}] 角色: {role}")
        
        if hasattr(content, 'parts'):
            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    text = part.text[:100] + "..." if len(part.text) > 100 else part.text
                    print(f"    内容: {text}")
        elif hasattr(content, 'text'):
            text = content.text[:100] + "..." if len(content.text) > 100 else content.text
            print(f"    内容: {text}")
    
    print(f"\n总对话轮次: {len(history)}")
    print("\n提示：聊天历史可以用于：")
    print("- 分析对话内容")
    print("- 导出对话记录")
    print("- 实现对话总结功能（将在后续示例中演示）")


if __name__ == "__main__":
    try:
        # 示例 1: 基本聊天会话
        demonstrate_chat_sessions()
        
        # 示例 2: 上下文保持
        demonstrate_context_retention()
        
        # 示例 3: 聊天历史访问
        demonstrate_chat_history_access()
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

