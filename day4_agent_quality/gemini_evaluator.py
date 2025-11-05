"""
Day 4 - 示例 3: 使用 Gemini 作为评估者
演示如何使用 Gemini 模型本身作为评估者，根据预定义的标准评估智能体的输出。
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
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
class EvaluationCriteria:
    """评估标准"""
    relevance: str = "答案是否与问题相关"
    completeness: str = "答案是否完整回答了问题"
    accuracy: str = "答案是否准确"
    helpfulness: str = "答案是否对用户有帮助"


@dataclass
class GeminiEvaluationResult:
    """Gemini评估结果"""
    score: float  # 0-10分
    reasoning: str
    criteria_scores: Dict[str, float]
    feedback: str


class GeminiEvaluator:
    """使用Gemini作为评估者"""
    
    def __init__(self, model_name: str = None, criteria: Optional[EvaluationCriteria] = None):
        if model_name is None:
            model_name = get_default_model()
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.criteria = criteria or EvaluationCriteria()
    
    def evaluate(
        self,
        query: str,
        expected_answer: Optional[str],
        actual_answer: str,
        context: Optional[str] = None
    ) -> GeminiEvaluationResult:
        """
        使用Gemini评估智能体的回答
        
        Args:
            query: 用户查询
            expected_answer: 期望的答案（可选）
            actual_answer: 智能体的实际回答
            context: 额外上下文信息（可选）
            
        Returns:
            GeminiEvaluationResult: 评估结果
        """
        # 构建评估提示
        prompt = self._build_evaluation_prompt(
            query, expected_answer, actual_answer, context
        )
        
        try:
            response = self.model.generate_content(prompt)
            
            # 解析响应
            if response.candidates and response.candidates[0].content.parts:
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text += part.text
            else:
                text = response.text if hasattr(response, 'text') else str(response)
            
            return self._parse_evaluation_response(text)
            
        except Exception as e:
            return GeminiEvaluationResult(
                score=0.0,
                reasoning=f"评估失败: {str(e)}",
                criteria_scores={},
                feedback="无法进行评估"
            )
    
    def _build_evaluation_prompt(
        self,
        query: str,
        expected_answer: Optional[str],
        actual_answer: str,
        context: Optional[str]
    ) -> str:
        """构建评估提示"""
        prompt = f"""你是一个AI智能体评估专家。请评估以下智能体的回答质量。

用户查询: {query}

智能体的回答:
{actual_answer}
"""
        
        if expected_answer:
            prompt += f"\n期望的答案（参考）:\n{expected_answer}\n"
        
        if context:
            prompt += f"\n上下文信息:\n{context}\n"
        
        prompt += f"""
请根据以下标准评估回答质量，并给出0-10分的评分：

1. **相关性 (Relevance)**: {self.criteria.relevance}
2. **完整性 (Completeness)**: {self.criteria.completeness}
3. **准确性 (Accuracy)**: {self.criteria.accuracy}
4. **有用性 (Helpfulness)**: {self.criteria.helpfulness}

请以JSON格式返回评估结果，格式如下：
{{
    "overall_score": <0-10之间的分数>,
    "criteria_scores": {{
        "relevance": <0-10>,
        "completeness": <0-10>,
        "accuracy": <0-10>,
        "helpfulness": <0-10>
    }},
    "reasoning": "<评估理由，说明为什么给出这个分数>",
    "feedback": "<改进建议>"
}}

只返回JSON，不要其他内容："""
        
        return prompt
    
    def _parse_evaluation_response(self, response_text: str) -> GeminiEvaluationResult:
        """解析评估响应"""
        try:
            # 尝试提取JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return GeminiEvaluationResult(
                    score=float(data.get('overall_score', 0)),
                    reasoning=data.get('reasoning', ''),
                    criteria_scores=data.get('criteria_scores', {}),
                    feedback=data.get('feedback', '')
                )
        except Exception as e:
            pass
        
        # 如果解析失败，尝试提取分数
        import re
        score_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
        score = float(score_match.group(1)) if score_match else 5.0
        
        return GeminiEvaluationResult(
            score=score,
            reasoning="无法解析详细评估，仅提取了分数",
            criteria_scores={},
            feedback=response_text[:500]  # 使用前500字符作为反馈
        )
    
    def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[GeminiEvaluationResult]:
        """批量评估"""
        results = []
        for i, eval_data in enumerate(evaluations, 1):
            print(f"评估 {i}/{len(evaluations)}...")
            result = self.evaluate(
                query=eval_data.get('query', ''),
                expected_answer=eval_data.get('expected_answer'),
                actual_answer=eval_data.get('actual_answer', ''),
                context=eval_data.get('context')
            )
            results.append(result)
        return results


def demonstrate_gemini_evaluator():
    """演示Gemini评估者"""
    print("=" * 60)
    print("Gemini作为评估者示例")
    print("=" * 60)
    print()
    
    # 创建评估者
    evaluator = GeminiEvaluator()
    
    # 测试用例
    test_cases = [
        {
            "query": "请帮我计算 25 × 4 + 100 ÷ 5",
            "expected_answer": "答案是120",
            "actual_answer": "计算结果: 25 * 4 + 100 / 5 = 120.0",
            "context": "这是一个数学计算问题"
        },
        {
            "query": "我今天有什么日程安排？",
            "expected_answer": "今天有团队会议和代码审查",
            "actual_answer": "今天的日程: 10:00 团队会议, 14:00 代码审查",
            "context": "用户询问今天的日程"
        },
        {
            "query": "Python是什么？",
            "expected_answer": None,  # 没有标准答案
            "actual_answer": "Python是一种编程语言，语法简洁，适合初学者。",
            "context": "用户询问Python的定义"
        },
        {
            "query": "请计算 100 + 200",
            "expected_answer": "300",
            "actual_answer": "我不确定如何计算，你可以使用计算器。",
            "context": "这是一个简单的计算问题"
        }
    ]
    
    print("开始评估智能体的回答...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"测试用例 {i}")
        print(f"{'='*60}")
        print(f"查询: {test_case['query']}")
        print(f"回答: {test_case['actual_answer']}")
        
        result = evaluator.evaluate(
            query=test_case['query'],
            expected_answer=test_case.get('expected_answer'),
            actual_answer=test_case['actual_answer'],
            context=test_case.get('context')
        )
        
        print(f"\n评估结果:")
        print(f"  总体评分: {result.score:.1f}/10")
        print(f"  评估理由: {result.reasoning[:200]}...")
        if result.criteria_scores:
            print(f"  标准评分:")
            for criterion, score in result.criteria_scores.items():
                print(f"    - {criterion}: {score:.1f}/10")
        print(f"  反馈: {result.feedback[:200]}...")
        print()
    
    print("=" * 60)
    print("评估完成")
    print("=" * 60)
    print("\n分析：")
    print("- Gemini能够理解问题并评估回答质量")
    print("- 可以给出多维度的评分（相关性、完整性、准确性、有用性）")
    print("- 提供详细的反馈和改进建议")
    print("- 可以用于自动化评估智能体的性能")


def demonstrate_custom_criteria():
    """演示自定义评估标准"""
    print("\n" + "=" * 60)
    print("自定义评估标准示例")
    print("=" * 60)
    print()
    
    # 定义自定义标准
    custom_criteria = EvaluationCriteria(
        relevance="答案是否直接回答了用户的问题",
        completeness="答案是否包含了所有必要的信息",
        accuracy="答案中的事实是否准确无误",
        helpfulness="答案是否提供了可操作的指导"
    )
    
    evaluator = GeminiEvaluator(criteria=custom_criteria)
    
    test_case = {
        "query": "如何学习Python？",
        "actual_answer": "学习Python可以从基础语法开始，然后学习常用库如NumPy和Pandas。建议通过实践项目来巩固知识。",
        "context": "用户想要学习Python编程"
    }
    
    print(f"查询: {test_case['query']}")
    print(f"回答: {test_case['actual_answer']}")
    
    result = evaluator.evaluate(
        query=test_case['query'],
        expected_answer=None,
        actual_answer=test_case['actual_answer'],
        context=test_case['context']
    )
    
    print(f"\n评估结果: {result.score:.1f}/10")
    print(f"详细评分: {json.dumps(result.criteria_scores, ensure_ascii=False, indent=2)}")
    print(f"反馈: {result.feedback}")


if __name__ == "__main__":
    try:
        # 示例 1: 基本评估
        demonstrate_gemini_evaluator()
        
        # 示例 2: 自定义标准
        demonstrate_custom_criteria()
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

