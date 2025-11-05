"""
模型工具函数：用于获取和验证可用的 Gemini 模型
（与 day1 和 day2 共享相同的模型工具函数）
"""

import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# 加载环境变量（从项目根目录）
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# 配置 Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def get_available_models():
    """获取所有可用的 Gemini 模型列表"""
    try:
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available.append(model.name)
        return available
    except Exception as e:
        print(f"获取模型列表时出错: {e}")
        return []

def get_model_name(preferred_models: list = None) -> str:
    """
    获取可用的模型名称
    
    Args:
        preferred_models: 优先使用的模型列表，按优先级排序
        
    Returns:
        可用的模型名称
    """
    if preferred_models is None:
        # 默认优先级：新版本优先
        preferred_models = [
            'gemini-1.5-flash',      # 更快的模型，优先使用
            'gemini-1.5-pro',        # 更强大的模型
            'gemini-pro',            # 旧版本（可能已废弃）
        ]
    
    # 首先尝试列出所有可用模型，然后选择
    try:
        available_models = get_available_models()
        if available_models:
            # 清理模型名称，提取基础名称
            available_clean = [m.replace('models/', '') for m in available_models]
            
            # 按优先级选择
            for preferred in preferred_models:
                if preferred in available_clean:
                    print(f"使用模型: {preferred}")
                    return preferred
            
            # 如果优先模型不在列表中，使用第一个可用模型
            if available_clean:
                print(f"使用可用模型: {available_clean[0]}")
                return available_clean[0]
    except Exception as e:
        print(f"获取模型列表时出错: {e}")
    
    # 如果无法列出模型，尝试直接创建模型实例
    print("\n尝试直接使用模型...")
    for model_name in preferred_models:
        try:
            model = genai.GenerativeModel(model_name)
            print(f"模型 {model_name} 可用")
            return model_name
        except Exception as e:
            continue
    
    # 如果所有方法都失败，返回默认值并提示用户
    print("\n警告: 无法自动检测可用模型，使用默认值 'gemini-1.5-flash'")
    print("如果遇到错误，请手动检查可用模型:")
    print("  python model_utils.py")
    return 'gemini-1.5-flash'

# 全局变量：缓存模型名称
_cached_model_name = None

def get_default_model() -> str:
    """获取默认模型名称（带缓存）"""
    global _cached_model_name
    if _cached_model_name is None:
        _cached_model_name = get_model_name()
    return _cached_model_name

if __name__ == "__main__":
    # 测试脚本：列出所有可用模型
    print("正在获取可用模型列表...\n")
    models = get_available_models()
    
    if models:
        print("可用的模型:")
        for model in models:
            print(f"  - {model}")
    else:
        print("无法获取模型列表。请检查 API key 是否正确配置。")
    
    print(f"\n推荐的模型: {get_default_model()}")

