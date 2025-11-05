"""
Day 3 - 示例 3: 上下文管理（对话总结）
演示如何总结冗长的对话记录以优化上下文窗口的使用。
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


class ConversationSummarizer:
    """对话总结器"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_default_model()
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def summarize_conversation(self, conversation_history: list, max_summary_length: int = 500):
        """
        总结对话历史
        
        Args:
            conversation_history: 对话历史列表
            max_summary_length: 最大总结长度
            
        Returns:
            总结文本
        """
        # 构建对话文本
        conversation_text = ""
        for i, content in enumerate(conversation_history):
            role = "用户" if (hasattr(content, 'role') and content.role == 'user') else "助手"
            
            # 提取文本内容
            text = ""
            if hasattr(content, 'parts'):
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        text += part.text
            elif hasattr(content, 'text'):
                text = content.text
            
            if text:
                conversation_text += f"{role}: {text}\n\n"
        
        # 构建总结提示
        prompt = f"""请总结以下对话的主要内容，保留关键信息和用户需求。

对话内容：
{conversation_text}

请提供一个简洁的总结（不超过{max_summary_length}字），包括：
1. 对话的主要主题
2. 用户的关键需求或问题
3. 重要的决定或结论
4. 需要记住的重要信息

总结："""
        
        try:
            response = self.model.generate_content(prompt)
            
            if response.candidates and response.candidates[0].content.parts:
                summary = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        summary += part.text
                return summary.strip()
            else:
                return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            print(f"总结生成失败: {e}")
            return f"对话总结失败: {str(e)}"
    
    def summarize_chat_history(self, chat, max_summary_length: int = 500):
        """
        总结聊天会话的历史
        
        Args:
            chat: Gemini ChatSession 对象
            max_summary_length: 最大总结长度
            
        Returns:
            总结文本
        """
        if not hasattr(chat, 'history'):
            return "无法访问聊天历史"
        
        history = chat.history
        if not history:
            return "聊天历史为空"
        
        return self.summarize_conversation(history, max_summary_length)


def demonstrate_conversation_summarization():
    """演示对话总结功能"""
    print("=" * 60)
    print("上下文管理示例 - 对话总结")
    print("=" * 60)
    print()
    
    model_name = get_default_model()
    model = genai.GenerativeModel(model_name)
    chat = model.start_chat()
    summarizer = ConversationSummarizer()
    
    # 创建一段较长的对话
    print("创建一段较长的对话...")
    print("-" * 60)
    
    long_conversation = [
        "我想了解Python编程语言。",
        "Python是一种高级编程语言，非常适合初学者。它语法简洁，功能强大。",
        "Python有哪些主要应用领域？",
        "Python广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。比如Django和Flask用于Web开发，NumPy和Pandas用于数据分析。",
        "我想学习数据科学，应该从哪里开始？",
        "对于数据科学，我建议从NumPy和Pandas开始，然后是数据可视化库如Matplotlib和Seaborn。",
        "我还需要了解机器学习吗？",
        "是的，机器学习是数据科学的重要组成部分。你可以从Scikit-learn开始，它提供了很多经典的机器学习算法。",
        "请为我制定一个学习计划。",
        "我建议的学习计划：1) 先掌握Python基础；2) 学习NumPy和Pandas；3) 学习数据可视化；4) 学习机器学习基础；5) 实践项目。每个阶段大约需要2-4周。"
    ]
    
    # 模拟对话
    for i, message in enumerate(long_conversation):
        if i % 2 == 0:  # 用户消息
            print(f"\n[用户] {message}")
            try:
                response = chat.send_message(message)
                
                # 显示响应
                if response.candidates and response.candidates[0].content.parts:
                    text = ""
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            text += part.text
                    if text:
                        print(f"[助手] {text[:150]}...")
            except Exception as e:
                print(f"错误: {e}")
    
    # 显示对话历史长度
    history = chat.history
    print(f"\n对话历史包含 {len(history)} 条消息")
    
    # 总结对话
    print("\n" + "=" * 60)
    print("正在生成对话总结...")
    print("-" * 60)
    
    summary = summarizer.summarize_chat_history(chat)
    print(f"\n对话总结：\n{summary}")
    
    print("\n" + "=" * 60)
    print("分析：")
    print("- 总结保留了对话的关键信息（Python学习、数据科学路径、学习计划）")
    print("- 总结大大减少了文本长度，节省了上下文窗口")
    print("- 可以使用总结作为新的对话起点，继续对话而不丢失重要信息")


def demonstrate_context_optimization():
    """演示上下文窗口优化"""
    print("\n" + "=" * 60)
    print("上下文窗口优化示例")
    print("=" * 60)
    print()
    
    model_name = get_default_model()
    summarizer = ConversationSummarizer()
    
    # 模拟一个非常长的对话
    print("模拟一个包含20轮对话的长对话...")
    
    # 创建模拟的对话历史
    simulated_history = []
    topics = ["Python", "机器学习", "Web开发", "数据科学", "AI"]
    
    for i in range(20):
        role = "user" if i % 2 == 0 else "model"
        topic = topics[i % len(topics)]
        message = f"关于{topic}的问题{i+1}"
        
        # 创建简化的内容对象
        class SimpleContent:
            def __init__(self, role, text):
                self.role = role
                self.text = text
                self.parts = [SimplePart(text)]
        
        class SimplePart:
            def __init__(self, text):
                self.text = text
        
        simulated_history.append(SimpleContent(role, message))
    
    print(f"原始对话包含 {len(simulated_history)} 条消息")
    
    # 总结对话
    print("\n正在总结对话...")
    summary = summarizer.summarize_conversation(simulated_history)
    
    print(f"\n总结（{len(summary)} 字符）:")
    print(summary)
    
    print("\n" + "=" * 60)
    print("优化效果：")
    print(f"- 原始对话: {len(simulated_history)} 条消息")
    print(f"- 总结长度: {len(summary)} 字符")
    print(f"- 可以显著减少上下文窗口的使用")
    print(f"- 适合用于长期对话的上下文管理")


def demonstrate_incremental_summarization():
    """演示增量总结策略"""
    print("\n" + "=" * 60)
    print("增量总结策略示例")
    print("=" * 60)
    print()
    
    model_name = get_default_model()
    model = genai.GenerativeModel(model_name)
    chat = model.start_chat()
    summarizer = ConversationSummarizer()
    
    # 模拟多轮对话，定期总结
    print("模拟对话，每5轮进行一次总结...")
    print("-" * 60)
    
    conversation_segments = [
        ["我想学习编程", "推荐Python", "Python有什么优势？", "语法简洁，功能强大"],
        ["我想做Web开发", "推荐Django", "Django难学吗？", "相对容易，文档完善"],
        ["我还想学前端", "推荐React", "React和Vue哪个好？", "各有优势，React更流行"]
    ]
    
    summary_context = ""  # 累积的总结
    
    for segment_idx, segment in enumerate(conversation_segments, 1):
        print(f"\n--- 对话段 {segment_idx} ---")
        
        for msg in segment:
            if segment.index(msg) % 2 == 0:
                print(f"[用户] {msg}")
                try:
                    response = chat.send_message(msg)
                except:
                    pass
        
        # 每段结束后总结
        if len(chat.history) >= 4:
            current_summary = summarizer.summarize_chat_history(chat)
            summary_context += f"\n对话段{segment_idx}总结: {current_summary}"
            print(f"\n[总结] {current_summary[:100]}...")
    
    print("\n" + "=" * 60)
    print("累积总结：")
    print(summary_context)
    
    print("\n分析：")
    print("- 增量总结策略可以管理超长对话")
    print("- 每段对话都可以被总结和保存")
    print("- 总结可以用于后续对话的上下文")


if __name__ == "__main__":
    try:
        # 示例 1: 基本对话总结
        demonstrate_conversation_summarization()
        
        # 示例 2: 上下文优化
        demonstrate_context_optimization()
        
        # 示例 3: 增量总结
        demonstrate_incremental_summarization()
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

