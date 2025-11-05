"""
Day 5: 扩展性与成本优化
演示缓存、异步处理和模型选择策略
"""

import os
import time
import hashlib
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from functools import lru_cache
from cachetools import TTLCache, LRUCache
import google.generativeai as genai
from dotenv import load_dotenv

# 加载环境变量
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 配置 Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

from model_utils import get_model_for_task, get_default_model

class ResponseCache:
    """响应缓存：缓存常见查询的响应"""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存有效期（秒）
        """
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, query: str, model_name: str) -> str:
        """生成缓存键"""
        key_string = f"{model_name}:{query}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, model_name: str) -> Optional[str]:
        """从缓存获取响应"""
        key = self._get_key(query, model_name)
        result = self.cache.get(key)
        if result:
            self.hits += 1
            return result
        else:
            self.misses += 1
            return None
    
    def set(self, query: str, model_name: str, response: str):
        """设置缓存"""
        key = self._get_key(query, model_name)
        self.cache[key] = response
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache)
        }

class AsyncAgent:
    """异步智能体：支持并发处理多个请求"""
    
    def __init__(self, model_name: Optional[str] = None):
        """初始化异步智能体"""
        if model_name is None:
            model_name = get_model_for_task("simple")
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        print(f"异步智能体已初始化，使用模型: {model_name}")
    
    async def process_async(self, query: str) -> Dict[str, Any]:
        """
        异步处理查询
        
        Args:
            query: 用户查询
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 在线程池中执行同步的 API 调用
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(query)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "answer": response.text if response.text else "无法生成回答",
            "latency_ms": round(latency_ms, 2),
            "model": self.model_name
        }
    
    async def process_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理查询
        
        Args:
            queries: 查询列表
            
        Returns:
            结果列表
        """
        tasks = [self.process_async(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results

class OptimizedAgent:
    """优化的智能体：结合缓存、模型选择和异步处理"""
    
    def __init__(self):
        """初始化优化智能体"""
        self.cache = ResponseCache(max_size=200, ttl=3600)
        self.simple_model = genai.GenerativeModel(get_model_for_task("simple"))
        self.complex_model = genai.GenerativeModel(get_model_for_task("complex"))
        self.simple_model_name = get_model_for_task("simple")
        self.complex_model_name = get_model_for_task("complex")
        print("优化智能体已初始化")
    
    def _should_use_complex_model(self, query: str) -> bool:
        """判断是否应该使用复杂模型"""
        complex_keywords = [
            "分析", "解释", "详细", "比较", "评估", "总结",
            "analyze", "explain", "compare", "evaluate", "summarize"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in complex_keywords)
    
    def process(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        处理查询（带优化）
        
        Args:
            query: 用户查询
            use_cache: 是否使用缓存
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 选择合适的模型
        use_complex = self._should_use_complex_model(query)
        model_name = self.complex_model_name if use_complex else self.simple_model_name
        model = self.complex_model if use_complex else self.simple_model
        
        # 检查缓存
        if use_cache:
            cached_response = self.cache.get(query, model_name)
            if cached_response:
                latency_ms = (time.time() - start_time) * 1000
                return {
                    "query": query,
                    "answer": cached_response,
                    "latency_ms": round(latency_ms, 2),
                    "model": model_name,
                    "cached": True,
                    "cost_optimized": True
                }
        
        # 生成响应
        response = model.generate_content(query)
        answer = response.text if response.text else "无法生成回答"
        
        # 存入缓存
        if use_cache:
            self.cache.set(query, model_name, answer)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "answer": answer,
            "latency_ms": round(latency_ms, 2),
            "model": model_name,
            "cached": False,
            "cost_optimized": not use_complex
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()

def demonstrate_caching():
    """演示缓存效果"""
    print("=" * 60)
    print("缓存优化演示")
    print("=" * 60)
    
    agent = OptimizedAgent()
    queries = [
        "什么是人工智能？",
        "什么是人工智能？",  # 重复查询
        "请解释机器学习",
        "什么是人工智能？",  # 再次重复
    ]
    
    print("\n处理查询（启用缓存）...\n")
    for i, query in enumerate(queries, 1):
        result = agent.process(query, use_cache=True)
        print(f"查询 {i}: {query}")
        print(f"  回答: {result['answer'][:50]}...")
        print(f"  延迟: {result['latency_ms']}ms")
        print(f"  缓存命中: {result['cached']}")
        print(f"  模型: {result['model']}")
        print()
    
    stats = agent.get_cache_stats()
    print("缓存统计:")
    print(f"  命中次数: {stats['hits']}")
    print(f"  未命中次数: {stats['misses']}")
    print(f"  命中率: {stats['hit_rate']}%")
    print(f"  缓存大小: {stats['cache_size']}")

async def demonstrate_async():
    """演示异步处理"""
    print("\n" + "=" * 60)
    print("异步处理演示")
    print("=" * 60)
    
    agent = AsyncAgent()
    queries = [
        "什么是人工智能？",
        "请解释机器学习",
        "深度学习的优势是什么？",
        "自然语言处理的应用有哪些？",
    ]
    
    print(f"\n并发处理 {len(queries)} 个查询...\n")
    
    start_time = time.time()
    results = await agent.process_batch(queries)
    total_time = time.time() - start_time
    
    print("处理结果:")
    for i, result in enumerate(results, 1):
        print(f"查询 {i}: {result['query']}")
        print(f"  回答: {result['answer'][:50]}...")
        print(f"  延迟: {result['latency_ms']}ms")
        print()
    
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均延迟: {sum(r['latency_ms'] for r in results) / len(results):.2f}ms")
    
    # 对比串行处理时间
    print("\n对比：串行处理估计时间:")
    estimated_serial_time = sum(r['latency_ms'] for r in results) / 1000
    print(f"  估计串行时间: {estimated_serial_time:.2f} 秒")
    print(f"  加速比: {estimated_serial_time / total_time:.2f}x")

def demonstrate_model_selection():
    """演示模型选择策略"""
    print("\n" + "=" * 60)
    print("模型选择优化演示")
    print("=" * 60)
    
    agent = OptimizedAgent()
    
    # 简单查询（应该使用 Flash）
    simple_queries = [
        "什么是AI？",
        "1+1等于多少？",
        "今天天气怎么样？",
    ]
    
    # 复杂查询（应该使用 Pro）
    complex_queries = [
        "请详细分析人工智能和机器学习的关系",
        "请比较深度学习和传统机器学习的优缺点",
        "请总结自然语言处理的发展历程",
    ]
    
    print("\n简单查询（应使用 Flash 模型）:")
    for query in simple_queries:
        result = agent.process(query, use_cache=False)
        print(f"  查询: {query}")
        print(f"  模型: {result['model']}")
        print(f"  成本优化: {result['cost_optimized']}")
        print()
    
    print("复杂查询（应使用 Pro 模型）:")
    for query in complex_queries:
        result = agent.process(query, use_cache=False)
        print(f"  查询: {query}")
        print(f"  模型: {result['model']}")
        print(f"  成本优化: {result['cost_optimized']}")
        print()

def demonstrate_cost_optimization():
    """演示成本优化策略"""
    print("\n" + "=" * 60)
    print("成本优化策略总结")
    print("=" * 60)
    
    print("""
1. 缓存策略:
   - 缓存常见查询的响应
   - 减少重复的 API 调用
   - 显著降低成本和延迟

2. 模型选择:
   - 简单任务使用 Flash 模型（更快、更便宜）
   - 复杂任务使用 Pro 模型（更准确、更强大）
   - 根据查询复杂度自动选择

3. 异步处理:
   - 并发处理多个请求
   - 提高吞吐量
   - 减少总体响应时间

4. 批量处理:
   - 批量处理相似查询
   - 减少 API 调用开销
   - 提高效率

5. 延迟加载:
   - 只在需要时初始化模型
   - 减少内存占用
   - 降低启动成本
    """)

async def main():
    """主函数"""
    print("=" * 60)
    print("扩展性与成本优化演示")
    print("=" * 60)
    
    # 1. 缓存演示
    demonstrate_caching()
    
    # 2. 异步处理演示
    await demonstrate_async()
    
    # 3. 模型选择演示
    demonstrate_model_selection()
    
    # 4. 成本优化策略
    demonstrate_cost_optimization()

if __name__ == "__main__":
    asyncio.run(main())

