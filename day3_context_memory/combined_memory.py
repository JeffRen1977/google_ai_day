"""
Day 3 - 综合示例: 结合短期记忆和长期记忆
演示如何在一个智能体中同时使用短期记忆（聊天会话）和长期记忆（RAG）。
"""

import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from model_utils import get_default_model

# 导入 RAG 相关类
try:
    from rag_memory import RAGMemory, create_sample_knowledge_base
except ImportError:
    # 如果导入失败，定义简化版本
    class RAGMemory:
        def __init__(self, collection_name: str = "knowledge_base"):
            self.collection_name = collection_name
            self.collection = None
            print("警告: 使用简化版 RAGMemory")
        
        def add_knowledge(self, documents: list):
            print(f"添加 {len(documents)} 个文档到知识库（简化版）")
        
        def search(self, query: str, n_results: int = 3):
            return []
    
    def create_sample_knowledge_base():
        return [
            "Python是一种高级编程语言。",
            "Gemini是Google的AI模型。",
            "RAG是检索增强生成技术。"
        ]

# 导入总结器
try:
    from context_summarization import ConversationSummarizer
except ImportError:
    # 如果导入失败，定义简化版本
    class ConversationSummarizer:
        def __init__(self, model_name: str = None):
            if model_name is None:
                from model_utils import get_default_model
                model_name = get_default_model()
            self.model = genai.GenerativeModel(model_name)
        
        def summarize_chat_history(self, chat):
            return "对话总结（简化版）"

# 加载环境变量（从项目根目录）
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 配置 Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 GEMINI_API_KEY")

genai.configure(api_key=api_key)


class HybridMemoryAgent:
    """结合短期和长期记忆的智能体"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_default_model()
        
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.chat = self.model.start_chat()
        
        # 初始化长期记忆（RAG）
        self.rag_memory = RAGMemory(collection_name="agent_knowledge_base")
        
        # 初始化总结器
        self.summarizer = ConversationSummarizer(model_name)
        
        # 对话长度阈值（超过此值进行总结）
        self.summary_threshold = 10
    
    def add_knowledge(self, documents: list):
        """向长期记忆添加知识"""
        self.rag_memory.add_knowledge(documents)
    
    def chat_with_memory(self, user_message: str, use_rag: bool = True):
        """
        使用短期和长期记忆进行对话
        
        Args:
            user_message: 用户消息
            use_rag: 是否使用RAG检索长期记忆
            
        Returns:
            助手响应
        """
        # 1. 如果启用RAG，先检索相关长期记忆
        context_from_rag = ""
        if use_rag and self.rag_memory.collection:
            relevant_docs = self.rag_memory.search(user_message, n_results=2)
            if relevant_docs:
                context_from_rag = "\n相关背景信息：\n" + "\n".join(relevant_docs)
        
        # 2. 检查对话长度，如果太长则总结
        if len(self.chat.history) > self.summary_threshold:
            print("\n[系统] 对话历史较长，正在生成总结...")
            summary = self.summarizer.summarize_chat_history(self.chat)
            print(f"[总结] {summary[:200]}...")
            
            # 可以选择使用总结重新开始对话，这里仅作演示
            # 实际应用中，可以将总结作为新的上下文
        
        # 3. 构建增强的消息
        enhanced_message = user_message
        if context_from_rag:
            enhanced_message = context_from_rag + "\n\n用户问题：" + user_message
        
        # 4. 发送消息
        try:
            response = self.chat.send_message(enhanced_message)
            
            # 提取响应文本
            if response.candidates and response.candidates[0].content.parts:
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text += part.text
                return text
            else:
                return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            return f"错误: {e}"
    
    def get_conversation_summary(self):
        """获取当前对话的总结"""
        return self.summarizer.summarize_chat_history(self.chat)


def demonstrate_hybrid_memory():
    """演示混合记忆系统"""
    print("=" * 60)
    print("综合示例 - 混合记忆智能体")
    print("=" * 60)
    print()
    
    # 创建智能体
    agent = HybridMemoryAgent()
    
    # 添加知识库（长期记忆）
    print("初始化长期记忆...")
    knowledge_base = create_sample_knowledge_base()
    agent.add_knowledge(knowledge_base)
    print()
    
    # 进行多轮对话
    print("开始对话（使用短期和长期记忆）：")
    print("-" * 60)
    
    conversations = [
        "你好，我想了解Python编程。",
        "Python有哪些主要应用？",
        "我想学习数据科学，应该从哪里开始？",
        "之前我们讨论过Python，还记得吗？",
        "请总结一下我们刚才的对话。"
    ]
    
    for i, message in enumerate(conversations, 1):
        print(f"\n[轮次 {i}] 用户: {message}")
        response = agent.chat_with_memory(message, use_rag=True)
        print(f"助手: {response[:200]}..." if len(response) > 200 else f"助手: {response}")
        print("-" * 60)
    
    # 显示对话总结
    print("\n" + "=" * 60)
    print("对话总结：")
    print("-" * 60)
    summary = agent.get_conversation_summary()
    print(summary)
    
    print("\n" + "=" * 60)
    print("系统特点：")
    print("- ✅ 短期记忆：记住当前对话中的所有信息")
    print("- ✅ 长期记忆：从知识库检索相关信息")
    print("- ✅ 上下文管理：自动总结长对话")
    print("- ✅ 智能结合：同时利用两种记忆类型")


if __name__ == "__main__":
    try:
        demonstrate_hybrid_memory()
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

