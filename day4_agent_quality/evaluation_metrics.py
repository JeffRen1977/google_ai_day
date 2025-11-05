"""
Day 4 - 示例 2: 评估指标
演示如何评估智能体的性能，包括成功率、延迟、准确性、工具调用有效性等指标。
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
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


@dataclass
class TestCase:
    """测试用例"""
    query: str
    expected_tool: Optional[str] = None
    expected_answer_contains: Optional[List[str]] = None
    expected_answer_type: Optional[str] = None  # "number", "text", "list"


@dataclass
class EvaluationResult:
    """评估结果"""
    test_case: TestCase
    success: bool
    latency_ms: float
    tool_called: Optional[str] = None
    actual_answer: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class AgentEvaluator:
    """智能体评估器"""
    
    def __init__(self, agent):
        self.agent = agent
        self.results: List[EvaluationResult] = []
    
    def evaluate(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """评估智能体性能"""
        print("=" * 60)
        print("开始评估智能体")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}/{len(test_cases)}: {test_case.query}")
            result = self._run_test_case(test_case)
            self.results.append(result)
            
            status = "✓" if result.success else "✗"
            print(f"{status} 成功: {result.success}, 延迟: {result.latency_ms:.2f}ms")
            if result.error:
                print(f"  错误: {result.error}")
        
        # 计算指标
        metrics = self._calculate_metrics()
        self._print_metrics(metrics)
        
        return metrics
    
    def _run_test_case(self, test_case: TestCase) -> EvaluationResult:
        """运行单个测试用例"""
        start_time = time.time()
        
        try:
            # 运行智能体
            answer = self.agent.run(test_case.query)
            latency_ms = (time.time() - start_time) * 1000
            
            # 检查结果
            success = self._check_success(test_case, answer)
            
            # 提取工具调用信息（如果logger可用）
            tool_called = None
            if hasattr(self.agent, 'logger'):
                summary = self.agent.logger.get_trace_summary()
                tool_calls = summary.get('tool_calls', [])
                if tool_calls:
                    tool_called = tool_calls[0].get('tool_name')
            
            return EvaluationResult(
                test_case=test_case,
                success=success,
                latency_ms=latency_ms,
                tool_called=tool_called,
                actual_answer=answer,
                metrics=self._calculate_case_metrics(test_case, answer, latency_ms)
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                test_case=test_case,
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def _check_success(self, test_case: TestCase, answer: str) -> bool:
        """检查测试用例是否成功"""
        success = True
        
        # 检查工具调用
        if test_case.expected_tool:
            # 这里简化处理，实际应该检查日志
            if test_case.expected_tool not in answer.lower():
                # 工具检查需要在日志中，这里仅作示例
                pass
        
        # 检查答案内容
        if test_case.expected_answer_contains:
            for keyword in test_case.expected_answer_contains:
                if keyword not in answer:
                    success = False
                    break
        
        # 检查答案类型
        if test_case.expected_answer_type:
            if test_case.expected_answer_type == "number":
                try:
                    # 尝试提取数字
                    import re
                    numbers = re.findall(r'\d+', answer)
                    if not numbers:
                        success = False
                except:
                    success = False
        
        return success
    
    def _calculate_case_metrics(self, test_case: TestCase, answer: str, latency_ms: float) -> Dict[str, Any]:
        """计算单个测试用例的指标"""
        return {
            "answer_length": len(answer),
            "latency_ms": latency_ms,
            "has_tool_call": hasattr(self.agent, 'logger') and self.agent.logger is not None
        }
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """计算总体指标"""
        if not self.results:
            return {}
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        
        latencies = [r.latency_ms for r in self.results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        
        # 工具调用有效性
        tool_calls = [r for r in self.results if r.tool_called]
        tool_call_count = len(tool_calls)
        tool_call_success = sum(1 for r in tool_calls if r.success)
        tool_call_effectiveness = (tool_call_success / tool_call_count * 100) if tool_call_count > 0 else 0
        
        return {
            "total_tests": total,
            "success_count": successful,
            "failed_count": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "average_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "tool_call_count": tool_call_count,
            "tool_call_effectiveness": tool_call_effectiveness
        }
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """打印指标"""
        print("\n" + "=" * 60)
        print("评估指标")
        print("=" * 60)
        print(f"总测试数: {metrics.get('total_tests', 0)}")
        print(f"成功数: {metrics.get('success_count', 0)}")
        print(f"失败数: {metrics.get('failed_count', 0)}")
        print(f"成功率: {metrics.get('success_rate', 0):.2f}%")
        print(f"\n延迟指标:")
        print(f"  平均延迟: {metrics.get('average_latency_ms', 0):.2f}ms")
        print(f"  最小延迟: {metrics.get('min_latency_ms', 0):.2f}ms")
        print(f"  最大延迟: {metrics.get('max_latency_ms', 0):.2f}ms")
        print(f"\n工具调用:")
        print(f"  工具调用次数: {metrics.get('tool_call_count', 0)}")
        print(f"  工具调用有效性: {metrics.get('tool_call_effectiveness', 0):.2f}%")
    
    def export_results(self, filename: str):
        """导出评估结果"""
        results_data = {
            "metrics": self._calculate_metrics(),
            "results": [asdict(r) for r in self.results]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已导出到: {filename}")


def create_golden_dataset() -> List[TestCase]:
    """创建黄金数据集（测试用例）"""
    return [
        TestCase(
            query="请帮我计算 25 × 4 + 100 ÷ 5",
            expected_tool="calculate",
            expected_answer_contains=["120"],
            expected_answer_type="number"
        ),
        TestCase(
            query="我今天有什么日程安排？",
            expected_tool="get_calendar_event",
            expected_answer_contains=["日程", "会议"]
        ),
        TestCase(
            query="北京今天的天气怎么样？",
            expected_tool="get_weather",
            expected_answer_contains=["北京", "天气"]
        ),
        TestCase(
            query="请计算 100 + 200",
            expected_tool="calculate",
            expected_answer_contains=["300"],
            expected_answer_type="number"
        ),
        TestCase(
            query="明天的日程是什么？",
            expected_tool="get_calendar_event",
            expected_answer_contains=["日程"]
        ),
    ]


def demonstrate_evaluation():
    """演示评估功能"""
    print("=" * 60)
    print("评估指标示例")
    print("=" * 60)
    print()
    
    # 导入带日志的智能体
    from logging_tracing import LoggedMultiToolAgent, AgentLogger
    
    # 创建智能体
    logger = AgentLogger()
    agent = LoggedMultiToolAgent(logger=logger)
    
    # 创建评估器
    evaluator = AgentEvaluator(agent)
    
    # 创建黄金数据集
    test_cases = create_golden_dataset()
    
    # 执行评估
    metrics = evaluator.evaluate(test_cases)
    
    # 导出结果
    evaluator.export_results("evaluation_results.json")
    
    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demonstrate_evaluation()
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

