"""
Day 1 - 示例 3: 传统 LLM vs 智能体对比演示
对比展示传统 LLM 应用和智能体的区别
"""

import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from simple_llm import simple_llm_query
from react_agent import ReActAgent
from model_utils import get_default_model

# 加载环境变量（从项目根目录）
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

def compare_llm_vs_agent():
    """对比传统 LLM 和智能体的区别"""
    
    print("\n" + "=" * 80)
    print("传统 LLM vs 智能体对比演示")
    print("=" * 80)
    
    # 测试用例
    test_cases = [
        {
            "query": "现在北京的时间是多少？",
            "description": "需要实时信息的任务（需要工具调用）"
        },
        {
            "query": "计算 15 × 23 + 45 ÷ 9 的结果",
            "description": "需要精确计算的任务（需要工具调用）"
        },
        {
            "query": "什么是人工智能？",
            "description": "知识性问题（不需要工具调用）"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试用例 {i}: {test_case['description']}")
        print(f"问题: {test_case['query']}")
        print(f"{'='*80}\n")
        
        # 传统 LLM 方式
        print("-" * 80)
        print("【传统 LLM 方式】")
        print("-" * 80)
        try:
            llm_response = simple_llm_query(test_case['query'])
            print(llm_response)
        except Exception as e:
            print(f"错误: {str(e)}")
        
        print("\n" + "-" * 80)
        print("特点:")
        print("✓ 单次调用，快速响应")
        print("✗ 无法调用外部工具")
        print("✗ 无法获取实时信息")
        print("✗ 计算可能不准确（基于训练数据）")
        print("-" * 80)
        
        # 智能体方式
        print("\n" + "-" * 80)
        print("【智能体方式 (ReAct)】")
        print("-" * 80)
        try:
            agent = ReActAgent()
            agent_response = agent.run(test_case['query'])
            print(f"\n最终响应: {agent_response}")
        except Exception as e:
            print(f"错误: {str(e)}")
        
        print("\n" + "-" * 80)
        print("特点:")
        print("✓ 可以调用外部工具")
        print("✓ 可以获取实时信息")
        print("✓ 精确计算")
        print("✓ 具备推理过程")
        print("✗ 需要多轮交互，响应时间较长")
        print("-" * 80)
        
        print("\n")
    
    # 总结
    print("\n" + "=" * 80)
    print("总结对比")
    print("=" * 80)
    print("""
传统 LLM:
- 适用场景: 文本生成、问答、内容创作
- 优点: 快速、简单、成本低
- 缺点: 无法调用工具、无实时信息、无状态管理

智能体 (Agent):
- 适用场景: 需要工具调用、实时信息、多步推理的任务
- 优点: 可以调用工具、获取实时信息、具备推理能力
- 缺点: 复杂度高、响应时间较长、成本较高

选择建议:
- 简单问答、文本生成 → 使用传统 LLM
- 需要工具调用、实时数据、复杂推理 → 使用智能体
    """)
    print("=" * 80)

if __name__ == "__main__":
    compare_llm_vs_agent()

