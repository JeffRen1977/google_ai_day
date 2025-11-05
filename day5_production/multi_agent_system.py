"""
Day 5: 多智能体系统 (Multi-Agent System, MAS)
演示 Agent2Agent (A2A) 协议：规划智能体（Planner）和执行智能体（Executor）
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
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

@dataclass
class Subtask:
    """子任务数据类"""
    id: str
    description: str
    tool: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None

@dataclass
class TaskPlan:
    """任务计划数据类"""
    task_id: str
    original_query: str
    subtasks: List[Subtask]
    status: str = "planning"  # planning, executing, completed, failed

class PlannerAgent:
    """规划智能体：接收任务并将其分解为子任务"""
    
    def __init__(self, model_name: Optional[str] = None):
        """初始化规划智能体"""
        if model_name is None:
            model_name = get_model_for_task("complex")  # 规划任务较复杂，使用 Pro 模型
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        print(f"规划智能体已初始化，使用模型: {model_name}")
    
    def plan(self, query: str) -> TaskPlan:
        """
        将任务分解为子任务
        
        Args:
            query: 用户查询
            
        Returns:
            任务计划
        """
        print(f"\n[规划智能体] 开始规划任务: {query}")
        
        # 构建规划提示
        planning_prompt = f"""你是一个任务规划智能体。请将以下任务分解为具体的子任务。

任务: {query}

请分析任务，并将其分解为可以执行的子任务。每个子任务应该：
1. 有清晰的描述
2. 指定需要的工具（如：calculate, search, weather, calendar 等）
3. 提供必要的参数

请以 JSON 格式返回，格式如下：
{{
    "subtasks": [
        {{
            "description": "子任务描述",
            "tool": "工具名称（如 calculate, search 等）",
            "parameters": {{"参数名": "参数值"}}
        }}
    ]
}}

如果任务很简单，可能只需要一个子任务。如果任务复杂，请分解为多个顺序执行的子任务。
只返回 JSON，不要添加其他解释。"""
        
        try:
            response = self.model.generate_content(planning_prompt)
            response_text = response.text.strip()
            
            # 清理响应文本（移除可能的 markdown 代码块）
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # 解析 JSON
            plan_data = json.loads(response_text)
            
            # 创建子任务列表
            subtasks = []
            for i, subtask_data in enumerate(plan_data.get("subtasks", []), 1):
                subtask = Subtask(
                    id=f"subtask_{i}",
                    description=subtask_data.get("description", ""),
                    tool=subtask_data.get("tool"),
                    parameters=subtask_data.get("parameters", {})
                )
                subtasks.append(subtask)
            
            # 创建任务计划
            task_plan = TaskPlan(
                task_id=f"task_{int(time.time())}",
                original_query=query,
                subtasks=subtasks,
                status="planning"
            )
            
            print(f"[规划智能体] 规划完成，共 {len(subtasks)} 个子任务")
            for i, subtask in enumerate(subtasks, 1):
                print(f"  {i}. {subtask.description} (工具: {subtask.tool})")
            
            return task_plan
            
        except json.JSONDecodeError as e:
            print(f"[规划智能体] JSON 解析错误: {e}")
            print(f"响应内容: {response_text}")
            # 返回一个简单的计划
            return TaskPlan(
                task_id=f"task_{int(time.time())}",
                original_query=query,
                subtasks=[Subtask(
                    id="subtask_1",
                    description=query,
                    tool=None
                )],
                status="planning"
            )
        except Exception as e:
            print(f"[规划智能体] 规划出错: {e}")
            raise

class ExecutorAgent:
    """执行智能体：接收子任务并使用工具执行"""
    
    def __init__(self, model_name: Optional[str] = None):
        """初始化执行智能体"""
        if model_name is None:
            model_name = get_model_for_task("simple")  # 执行任务较简单，使用 Flash 模型
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.tools = self._initialize_tools()
        print(f"执行智能体已初始化，使用模型: {model_name}")
    
    def _initialize_tools(self) -> Dict[str, callable]:
        """初始化可用工具"""
        return {
            "calculate": self._calculate,
            "search": self._search,
            "weather": self._weather,
            "calendar": self._calendar,
        }
    
    def _calculate(self, expression: str) -> str:
        """计算工具"""
        try:
            # 安全的表达式计算
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    def _search(self, query: str) -> str:
        """搜索工具（模拟）"""
        return f"搜索 '{query}' 的结果: [模拟搜索结果]"
    
    def _weather(self, location: str) -> str:
        """天气工具（模拟）"""
        return f"{location} 的天气: 晴天，温度 22°C"
    
    def _calendar(self, date: str, action: str = "check") -> str:
        """日历工具（模拟）"""
        if action == "check":
            return f"{date} 的日程: 无重要事项"
        elif action == "add":
            return f"已添加日程: {date}"
        else:
            return f"日历操作: {action} on {date}"
    
    def execute(self, subtask: Subtask) -> str:
        """
        执行子任务
        
        Args:
            subtask: 要执行的子任务
            
        Returns:
            执行结果
        """
        print(f"\n[执行智能体] 执行子任务: {subtask.description}")
        subtask.status = "in_progress"
        
        try:
            # 如果有指定工具，使用工具执行
            if subtask.tool and subtask.tool in self.tools:
                tool_func = self.tools[subtask.tool]
                params = subtask.parameters or {}
                
                # 根据工具类型调用
                if subtask.tool == "calculate":
                    result = tool_func(params.get("expression", ""))
                elif subtask.tool == "search":
                    result = tool_func(params.get("query", ""))
                elif subtask.tool == "weather":
                    result = tool_func(params.get("location", ""))
                elif subtask.tool == "calendar":
                    result = tool_func(
                        params.get("date", ""),
                        params.get("action", "check")
                    )
                else:
                    result = f"未知工具: {subtask.tool}"
            else:
                # 如果没有指定工具，使用 LLM 直接回答
                prompt = f"请回答以下问题: {subtask.description}"
                response = self.model.generate_content(prompt)
                result = response.text if response.text else "无法生成回答"
            
            subtask.status = "completed"
            subtask.result = result
            print(f"[执行智能体] 执行完成: {result[:100]}...")
            return result
            
        except Exception as e:
            subtask.status = "failed"
            subtask.result = f"执行错误: {str(e)}"
            print(f"[执行智能体] 执行失败: {e}")
            return subtask.result

class MultiAgentSystem:
    """多智能体系统：协调规划智能体和执行智能体"""
    
    def __init__(self):
        """初始化多智能体系统"""
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        print("\n多智能体系统已初始化")
    
    def process_task(self, query: str) -> Dict[str, Any]:
        """
        处理任务：规划 -> 执行 -> 汇总
        
        Args:
            query: 用户查询
            
        Returns:
            处理结果
        """
        print("=" * 60)
        print(f"处理任务: {query}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 步骤1: 规划
        task_plan = self.planner.plan(query)
        task_plan.status = "executing"
        
        # 步骤2: 执行所有子任务
        results = []
        for subtask in task_plan.subtasks:
            result = self.executor.execute(subtask)
            results.append({
                "subtask_id": subtask.id,
                "description": subtask.description,
                "result": result,
                "status": subtask.status
            })
        
        # 步骤3: 汇总结果
        task_plan.status = "completed"
        total_time = time.time() - start_time
        
        # 生成最终回答
        final_answer = self._generate_final_answer(query, results)
        
        return {
            "task_id": task_plan.task_id,
            "original_query": query,
            "subtasks_count": len(task_plan.subtasks),
            "subtasks": results,
            "final_answer": final_answer,
            "total_time_seconds": round(total_time, 2),
            "status": "completed"
        }
    
    def _generate_final_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """生成最终回答（可选：使用 LLM 汇总）"""
        # 简化版本：直接返回结果摘要
        if len(results) == 1:
            return results[0]["result"]
        else:
            summary = f"任务完成，共执行 {len(results)} 个子任务：\n"
            for i, result in enumerate(results, 1):
                summary += f"\n{i}. {result['description']}\n   结果: {result['result']}\n"
            return summary

def main():
    """主函数：演示多智能体系统"""
    print("=" * 60)
    print("多智能体系统演示 (Agent2Agent Protocol)")
    print("=" * 60)
    
    # 创建多智能体系统
    mas = MultiAgentSystem()
    
    # 测试用例
    test_queries = [
        "请帮我计算 25 × 4 + 100 ÷ 5",
        "请帮我计算 10 的平方，然后查询北京今天的天气",
        "请解释一下什么是人工智能",
    ]
    
    print("\n开始测试...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_queries)}")
        print(f"{'='*60}")
        
        result = mas.process_task(query)
        
        print(f"\n最终结果:")
        print(f"任务ID: {result['task_id']}")
        print(f"子任务数: {result['subtasks_count']}")
        print(f"总耗时: {result['total_time_seconds']} 秒")
        print(f"\n最终回答:\n{result['final_answer']}")
        print("\n" + "-"*60)
        
        # 等待一下再进行下一个测试
        if i < len(test_queries):
            time.sleep(1)

if __name__ == "__main__":
    main()

