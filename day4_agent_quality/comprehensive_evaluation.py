"""
Day 4 - 综合示例: 完整的评估系统
结合日志跟踪、评估指标和Gemini评估者，构建完整的智能体评估系统。
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import json
import google.generativeai as genai
from dotenv import load_dotenv
from model_utils import get_default_model

# 导入评估组件（使用相对导入，如果失败则使用简化版本）
try:
    from logging_tracing import LoggedMultiToolAgent, AgentLogger
    from evaluation_metrics import AgentEvaluator, TestCase, create_golden_dataset
    from gemini_evaluator import GeminiEvaluator
except ImportError:
    # 如果导入失败，提示用户
    print("警告: 无法导入评估组件，请确保所有文件都在同一目录下")
    raise

# 加载环境变量（从项目根目录）
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 配置 Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 GEMINI_API_KEY")

genai.configure(api_key=api_key)


class ComprehensiveEvaluator:
    """综合评估系统"""
    
    def __init__(self):
        # 创建带日志的智能体
        self.logger = AgentLogger()
        self.agent = LoggedMultiToolAgent(logger=self.logger)
        
        # 创建评估组件
        self.metrics_evaluator = AgentEvaluator(self.agent)
        self.gemini_evaluator = GeminiEvaluator()
    
    def evaluate_comprehensive(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """执行综合评估"""
        print("=" * 60)
        print("综合评估系统")
        print("=" * 60)
        print()
        
        # 1. 执行基础评估（成功率、延迟等）
        print("阶段1: 执行基础评估...")
        metrics = self.metrics_evaluator.evaluate(test_cases)
        
        # 2. 使用Gemini评估回答质量
        print("\n阶段2: 使用Gemini评估回答质量...")
        gemini_results = []
        
        for i, result in enumerate(self.metrics_evaluator.results):
            if result.actual_answer:
                print(f"评估回答 {i+1}/{len(self.metrics_evaluator.results)}...")
                gemini_result = self.gemini_evaluator.evaluate(
                    query=result.test_case.query,
                    expected_answer=None,
                    actual_answer=result.actual_answer
                )
                gemini_results.append(gemini_result)
        
        # 3. 计算综合评分
        print("\n阶段3: 计算综合评分...")
        avg_gemini_score = sum(r.score for r in gemini_results) / len(gemini_results) if gemini_results else 0
        
        # 4. 生成综合报告
        comprehensive_report = {
            "metrics": metrics,
            "gemini_evaluation": {
                "average_score": avg_gemini_score,
                "total_evaluations": len(gemini_results),
                "detailed_results": [
                    {
                        "query": self.metrics_evaluator.results[i].test_case.query,
                        "score": r.score,
                        "reasoning": r.reasoning,
                        "feedback": r.feedback
                    }
                    for i, r in enumerate(gemini_results)
                ]
            },
            "trace_summaries": [
                self.logger.get_trace_summary(f"test_{i+1}")
                for i in range(len(test_cases))
            ]
        }
        
        return comprehensive_report
    
    def print_report(self, report: Dict[str, Any]):
        """打印综合报告"""
        print("\n" + "=" * 60)
        print("综合评估报告")
        print("=" * 60)
        
        # 基础指标
        metrics = report.get("metrics", {})
        print("\n【基础指标】")
        print(f"  成功率: {metrics.get('success_rate', 0):.2f}%")
        print(f"  平均延迟: {metrics.get('average_latency_ms', 0):.2f}ms")
        print(f"  工具调用有效性: {metrics.get('tool_call_effectiveness', 0):.2f}%")
        
        # Gemini评估
        gemini_eval = report.get("gemini_evaluation", {})
        print("\n【Gemini质量评估】")
        print(f"  平均评分: {gemini_eval.get('average_score', 0):.1f}/10")
        print(f"  评估数量: {gemini_eval.get('total_evaluations', 0)}")
        
        # 详细结果
        print("\n【详细评估结果】")
        for i, detail in enumerate(gemini_eval.get('detailed_results', [])[:3], 1):
            print(f"\n  结果 {i}:")
            print(f"    查询: {detail['query'][:50]}...")
            print(f"    评分: {detail['score']:.1f}/10")
            print(f"    理由: {detail['reasoning'][:100]}...")
        
        # 跟踪摘要
        print("\n【跟踪摘要】")
        trace_summaries = report.get("trace_summaries", [])
        if trace_summaries:
            total_steps = sum(ts.get('total_steps', 0) for ts in trace_summaries)
            avg_duration = sum(ts.get('duration', 0) for ts in trace_summaries) / len(trace_summaries)
            print(f"  总步骤数: {total_steps}")
            print(f"  平均持续时间: {avg_duration:.2f}秒")
    
    def export_report(self, report: Dict[str, Any], filename: str):
        """导出综合报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n综合报告已导出到: {filename}")


def demonstrate_comprehensive_evaluation():
    """演示综合评估系统"""
    print("=" * 60)
    print("综合评估系统示例")
    print("=" * 60)
    print()
    
    # 创建综合评估器
    evaluator = ComprehensiveEvaluator()
    
    # 创建测试用例
    test_cases = create_golden_dataset()
    
    # 执行综合评估
    report = evaluator.evaluate_comprehensive(test_cases)
    
    # 打印报告
    evaluator.print_report(report)
    
    # 导出报告
    evaluator.export_report(report, "comprehensive_evaluation_report.json")
    
    # 导出日志
    evaluator.logger.export_logs("comprehensive_agent_logs.json")
    
    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)
    print("\n系统特点：")
    print("- ✅ 完整的日志跟踪（每个步骤）")
    print("- ✅ 详细的性能指标（成功率、延迟、工具调用）")
    print("- ✅ Gemini质量评估（相关性、完整性、准确性）")
    print("- ✅ 综合报告生成和导出")
    print("- ✅ 可用于持续改进智能体性能")


if __name__ == "__main__":
    try:
        demonstrate_comprehensive_evaluation()
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

