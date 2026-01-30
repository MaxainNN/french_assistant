"""
Ядро french_assistant.

Компоненты:
- FrenchAssistant: Главный класс ассистента
- AssistantConfig: Конфигурация системы
- load_config: Загрузка конфигурации из файла
"""

from .assistant import FrenchAssistant
from .config import (
    AssistantConfig,
    ModelConfig,
    VectorDBConfig,
    RAGConfig,
    SafetyConfig,
    load_config,
)

__all__ = [
    "FrenchAssistant",
    "AssistantConfig",
    "ModelConfig",
    "VectorDBConfig",
    "RAGConfig",
    "SafetyConfig",
    "load_config",
]
