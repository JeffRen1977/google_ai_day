# Day 3: 上下文工程：会话与内存管理 (Context Engineering: Sessions, Memory Management)

## 学习目标
- 理解短期记忆（会话历史）的实现和使用
- 学习长期记忆（RAG - 检索增强生成）的实现
- 掌握上下文管理技术，包括对话总结和上下文窗口优化

## 文件说明
- `chat_sessions.py`: 短期记忆示例 - 使用 Gemini Chat Sessions 进行多轮对话
- `rag_memory.py`: 长期记忆示例 - 使用 RAG 和向量数据库实现长期记忆
- `context_summarization.py`: 上下文管理示例 - 对话总结和上下文窗口优化
- `combined_memory.py`: 综合示例 - 结合短期和长期记忆的智能体

## 环境设置

1. 安装依赖：
```bash
cd day3_context_memory
pip install -r requirements.txt
```

**注意**：如果 `chromadb` 或 `sentence-transformers` 安装失败，程序会使用简化版本继续运行。

2. 配置 API Key：
```bash
# 在项目根目录（google_AI_day）创建 .env 文件（如果还没有）
# 填入你的 GEMINI_API_KEY:
# GEMINI_API_KEY=your_api_key_here
```

3. 运行示例：

**方式一：从 day3_context_memory 目录运行**
```bash
cd day3_context_memory
python3 chat_sessions.py              # 短期记忆示例
python3 rag_memory.py                 # 长期记忆示例
python3 context_summarization.py     # 上下文管理示例
python3 combined_memory.py            # 综合示例
```

**方式二：从项目根目录运行**
```bash
# 从项目根目录（google_AI_day）
python3 day3_context_memory/chat_sessions.py
python3 day3_context_memory/rag_memory.py
python3 day3_context_memory/context_summarization.py
python3 day3_context_memory/combined_memory.py
```

**注意**: 
- `.env` 文件应位于项目根目录（`google_AI_day/.env`），代码会自动从父目录加载环境变量。
- 代码会自动检测可用的 Gemini 模型（优先使用 `gemini-1.5-flash` 或 `gemini-1.5-pro`）。
- 在 macOS 上，如果 `python` 命令不可用，请使用 `python3`。

## 核心概念

### 1. 短期记忆（会话历史）

**概念**：
- 使用 Gemini API 的 `start_chat()` 方法创建聊天会话
- 聊天会话自动维护对话历史
- 模型能够记住当前对话中的所有信息

**实现**：
```python
model = genai.GenerativeModel(model_name)
chat = model.start_chat()
response = chat.send_message("用户消息")
# 后续消息会自动包含之前的对话历史
```

**特点**：
- 自动管理对话上下文
- 无需手动传递历史记录
- 适合短期对话场景

### 2. 长期记忆（RAG - 检索增强生成）

**概念**：
- 将外部知识存储在向量数据库中
- 使用语义搜索检索相关信息
- 将检索到的信息作为上下文提供给模型

**实现**：
```python
# 1. 创建向量数据库
rag = RAGMemory()
rag.add_knowledge(documents)

# 2. 检索相关文档
relevant_docs = rag.search(query)

# 3. 使用检索结果生成回答
answer = rag.query_with_rag(query)
```

**技术栈**：
- **向量数据库**：ChromaDB（轻量级，易于使用）
- **嵌入模型**：sentence-transformers（生成文本嵌入向量）
- **检索策略**：语义相似度搜索（余弦相似度）

**特点**：
- 可以存储大量知识
- 快速检索相关信息
- 适合知识库和文档检索场景

### 3. 上下文管理（对话总结）

**概念**：
- 当对话历史过长时，总结旧对话以节省上下文窗口
- 保留关键信息，丢弃细节
- 使用总结作为新的对话起点

**实现**：
```python
summarizer = ConversationSummarizer()
summary = summarizer.summarize_chat_history(chat)
```

**策略**：
1. **固定长度总结**：当对话达到一定长度时总结
2. **增量总结**：定期总结对话段落
3. **关键信息提取**：保留重要决定和需求

## 技术实现细节

### 向量数据库（ChromaDB）

**安装**：
```bash
pip install chromadb
```

**特点**：
- 轻量级，易于使用
- 支持持久化存储
- 自动管理向量索引
- 支持多种距离度量（余弦、欧氏距离等）

**使用**：
```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("knowledge_base")
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=ids
)
```

### 嵌入模型（Sentence Transformers）

**安装**：
```bash
pip install sentence-transformers
```

**特点**：
- 预训练模型，开箱即用
- 支持多语言
- 高质量语义表示
- 模型大小适中（适合本地运行）

**使用**：
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
```

### 简化版本（Fallback）

如果无法安装向量数据库或嵌入模型，程序会使用简化版本：
- 使用哈希函数生成模拟向量
- 使用内存存储（不持久化）
- 功能受限，但可以运行

## 最佳实践

### 1. 短期记忆使用场景
- ✅ 当前对话的上下文管理
- ✅ 多轮问答
- ✅ 需要记住用户偏好的场景
- ❌ 不适合存储大量历史信息

### 2. 长期记忆使用场景
- ✅ 知识库查询
- ✅ 文档检索
- ✅ 需要访问大量外部信息的场景
- ❌ 不适合频繁更新的信息

### 3. 上下文管理策略
- **阈值策略**：当对话长度超过阈值时总结
- **时间策略**：定期总结（如每10轮）
- **主题策略**：当话题改变时总结上一话题

### 4. 混合记忆系统
- 短期记忆：当前对话的详细信息
- 长期记忆：知识库和文档
- 总结：压缩的历史对话

## 测试状态

所有程序已通过测试，可以正常运行：

- ✅ `chat_sessions.py` - 短期记忆示例
- ✅ `rag_memory.py` - 长期记忆示例（简化版本，不依赖外部库也能运行）
- ✅ `context_summarization.py` - 上下文管理示例
- ✅ `combined_memory.py` - 综合示例

### 已知问题

- 如果 `chromadb` 或 `sentence-transformers` 未安装，程序会使用简化版本
- 简化版本的搜索效果可能不如完整版本
- 向量数据库的持久化数据存储在 `.chromadb` 目录中

## 示例输出

### 短期记忆示例输出：

```
============================================================
短期记忆示例 - 聊天会话（Chat Sessions）
============================================================

创建新的聊天会话...
[轮次 1] 用户: 我的名字是张三，我是一名软件工程师。
助手: 你好张三！很高兴认识你，软件工程师是一份很有前景的职业...
[对话历史统计] 消息数量: 2
```

### RAG 长期记忆示例输出：

```
============================================================
长期记忆示例 - RAG（检索增强生成）
============================================================

查询: Python是什么？
正在检索相关文档...
找到 3 个相关文档
正在生成回答...

回答: Python是一种高级编程语言，由Guido van Rossum在1991年首次发布...
```

### 上下文管理示例输出：

```
============================================================
上下文管理示例 - 对话总结
============================================================

对话历史包含 10 条消息
正在生成对话总结...

对话总结：
主要主题：Python学习和数据科学入门
用户需求：学习Python并了解数据科学路径
重要决定：制定了5阶段学习计划...
```

## 扩展学习

1. **高级 RAG 技术**：
   - 多跳检索（Multi-hop Retrieval）
   - 重排序（Re-ranking）
   - 混合搜索（Hybrid Search）

2. **上下文优化**：
   - 滑动窗口策略
   - 关键信息提取
   - 对话压缩算法

3. **向量数据库选择**：
   - Pinecone（云服务）
   - Weaviate（开源）
   - Qdrant（高性能）

4. **嵌入模型选择**：
   - OpenAI Embeddings
   - Cohere Embeddings
   - 自定义微调模型

