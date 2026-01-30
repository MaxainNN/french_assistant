"""
French Assistant - RAG-система для перевода с русского на французский.

Пакет предоставляет:
- FrenchAssistant: Главный класс для работы с ассистентом
- Модули безопасности, retrieval и RAG-улучшений

Быстрый старт:
    from french_assistant import FrenchAssistant

    assistant = FrenchAssistant()
    result = assistant.process_query("Переведи: Привет, как дела?")
    print(result["response"])

Модули:
- core: Ядро системы (FrenchAssistant, конфигурация)
- safety: Безопасность (SafetyFilter, HallucinationDetector)
- retrieval: Поиск (EnhancedRetriever, VectorStoreManager)
- enhancements: RAG-улучшения (CRAG, Self-RAG, CoVe)
- utils: Утилиты (TracingManager, логирование)
"""

__version__ = "1.0.0"
__author__ = "Maxim Kalugin"

from .core.assistant import FrenchAssistant
from .core.config import AssistantConfig, load_config, get_hf_token
from .safety import SafetyFilter, HallucinationDetector
from .retrieval import EnhancedRetriever, VectorStoreManager, QueryExpander
from .enhancements import SelfRAG, CorrectiveRAG, ChainOfVerification
from .utils import TracingManager, setup_logging

__all__ = [
    "FrenchAssistant",
    "AssistantConfig",
    "load_config",
    "get_hf_token",
    "SafetyFilter",
    "HallucinationDetector",
    "EnhancedRetriever",
    "VectorStoreManager",
    "QueryExpander",
    "SelfRAG",
    "CorrectiveRAG",
    "ChainOfVerification",
    "TracingManager",
    "setup_logging",
]
