"""
Day 3 - 示例 2: 长期记忆（RAG - 检索增强生成）
演示如何使用向量数据库实现长期记忆，让智能体能够检索和利用外部知识库。
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

# 尝试导入向量数据库库
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("警告: chromadb 未安装，将使用模拟的向量数据库")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装，将使用简化的嵌入方法")


class RAGMemory:
    """RAG 长期记忆系统"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.embedder = None
        self.client = None
        self.collection = None
        
        # 初始化嵌入模型
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # 使用轻量级的模型
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("已加载嵌入模型: all-MiniLM-L6-v2")
            except Exception as e:
                print(f"加载嵌入模型失败: {e}")
                self.embedder = None
        
        # 初始化向量数据库
        if CHROMADB_AVAILABLE:
            try:
                self.client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(Path(__file__).parent / ".chromadb")
                ))
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"已连接到向量数据库: {collection_name}")
            except Exception as e:
                print(f"初始化向量数据库失败: {e}")
                self.collection = None
    
    def _get_embedding(self, text: str):
        """获取文本的嵌入向量"""
        if self.embedder:
            return self.embedder.encode(text).tolist()
        else:
            # 简化的模拟嵌入（实际应用中应使用真实的嵌入模型）
            # 这里返回一个简单的向量表示
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # 将hash转换为128维向量（简化版）
            vector = [int(hash_hex[i:i+2], 16) / 255.0 for i in range(0, min(32, len(hash_hex)), 2)]
            # 填充到128维
            while len(vector) < 128:
                vector.extend(vector[:128 - len(vector)])
            return vector[:128]
    
    def add_knowledge(self, documents: list, metadatas: list = None, ids: list = None):
        """向知识库添加文档"""
        if not self.collection:
            print("向量数据库不可用，使用内存存储")
            return
        
        if not documents:
            return
        
        # 生成嵌入向量
        embeddings = [self._get_embedding(doc) for doc in documents]
        
        # 生成ID（如果没有提供）
        if not ids:
            import hashlib
            ids = [hashlib.md5(doc.encode()).hexdigest() for doc in documents]
        
        # 准备元数据
        if not metadatas:
            metadatas = [{"source": "manual"} for _ in documents]
        
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"已添加 {len(documents)} 个文档到知识库")
        except Exception as e:
            print(f"添加文档失败: {e}")
    
    def search(self, query: str, n_results: int = 3):
        """在知识库中搜索相关文档"""
        if not self.collection:
            print("向量数据库不可用，无法搜索")
            return []
        
        try:
            # 获取查询的嵌入向量
            query_embedding = self._get_embedding(query)
            
            # 搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # 提取文档
            documents = []
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
            
            return documents
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def query_with_rag(self, query: str, model_name: str = None):
        """使用 RAG 查询：检索相关文档并生成答案"""
        if model_name is None:
            model_name = get_default_model()
        
        model = genai.GenerativeModel(model_name)
        
        # 1. 检索相关文档
        print(f"\n查询: {query}")
        print("-" * 60)
        print("正在检索相关文档...")
        relevant_docs = self.search(query, n_results=3)
        
        if not relevant_docs:
            print("未找到相关文档，使用模型直接回答")
            response = model.generate_content(query)
            return response.text if hasattr(response, 'text') else str(response)
        
        print(f"找到 {len(relevant_docs)} 个相关文档")
        print("-" * 60)
        
        # 2. 构建上下文
        context = "\n\n".join([f"文档 {i+1}: {doc}" for i, doc in enumerate(relevant_docs)])
        
        # 3. 构建提示
        prompt = f"""基于以下上下文信息回答用户的问题。如果上下文信息不足以回答问题，请说明。

上下文信息：
{context}

用户问题：{query}

请基于上下文信息提供准确、完整的回答："""
        
        # 4. 生成回答
        print("正在生成回答...")
        response = model.generate_content(prompt)
        
        if response.candidates and response.candidates[0].content.parts:
            answer = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    answer += part.text
            return answer
        else:
            return response.text if hasattr(response, 'text') else str(response)


def create_sample_knowledge_base():
    """创建示例知识库"""
    knowledge_base = [
        "Python是一种高级编程语言，由Guido van Rossum在1991年首次发布。",
        "Python的设计哲学强调代码的可读性，特别是使用空格和缩进。",
        "Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。",
        "Python的标准库非常庞大，被称为'内置电池'哲学。",
        "Python广泛应用于Web开发、数据科学、人工智能、自动化等领域。",
        "Google的Gemini是一个多模态AI模型，能够理解和处理文本、图像、音频等多种输入。",
        "Gemini API提供了多种模型，包括Gemini Pro、Gemini Flash等不同版本。",
        "Gemini模型支持函数调用（Function Calling），可以让AI模型调用外部工具。",
        "RAG（检索增强生成）是一种结合信息检索和生成的技术，可以提高AI回答的准确性。",
        "向量数据库是存储和检索高维向量数据的高效方式，常用于语义搜索。"
    ]
    return knowledge_base


def demonstrate_rag_memory():
    """演示 RAG 长期记忆功能"""
    print("=" * 60)
    print("长期记忆示例 - RAG（检索增强生成）")
    print("=" * 60)
    print()
    
    # 创建 RAG 记忆系统
    rag = RAGMemory()
    
    # 创建示例知识库
    print("创建示例知识库...")
    knowledge_base = create_sample_knowledge_base()
    rag.add_knowledge(knowledge_base)
    print()
    
    # 测试查询
    queries = [
        "Python是什么？",
        "Gemini API有哪些特点？",
        "什么是RAG？",
        "向量数据库的用途是什么？"
    ]
    
    print("执行查询测试：")
    print("=" * 60)
    
    for query in queries:
        answer = rag.query_with_rag(query)
        print(f"\n回答: {answer}")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("分析：")
    print("- RAG系统能够从知识库中检索相关信息")
    print("- 模型基于检索到的上下文生成更准确的回答")
    print("- 这实现了长期记忆：即使知识库很大，也能快速找到相关信息")


def demonstrate_custom_knowledge_base():
    """演示自定义知识库"""
    print("\n" + "=" * 60)
    print("自定义知识库示例")
    print("=" * 60)
    print()
    
    # 创建新的知识库
    rag = RAGMemory(collection_name="company_policy")
    
    # 添加公司政策文档
    policies = [
        "公司工作时间为周一至周五，上午9点到下午6点。",
        "员工每年享有15天带薪年假。",
        "公司提供健康保险和401k退休计划。",
        "远程工作需要提前申请并获得批准。",
        "公司支持员工参加技术会议和培训。"
    ]
    
    rag.add_knowledge(policies)
    
    # 查询
    questions = [
        "公司的年假政策是什么？",
        "如何申请远程工作？",
        "公司有哪些员工福利？"
    ]
    
    for question in questions:
        answer = rag.query_with_rag(question)
        print(f"\n问题: {question}")
        print(f"回答: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    try:
        # 示例 1: 基本 RAG 功能
        demonstrate_rag_memory()
        
        # 示例 2: 自定义知识库
        demonstrate_custom_knowledge_base()
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

