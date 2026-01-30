"""
Модуль RAG-улучшений для french_assistant.

Компоненты:
- SelfRAG: Самооценка качества retrieval
- CorrectiveRAG: Коррекция результатов retrieval
- ChainOfVerification: Пошаговая верификация ответов
- RetrievalQuality: Enum для оценки качества
- VerificationResult: Результат верификации
"""

from .self_rag import SelfRAG, RetrievalQuality
from .crag import CorrectiveRAG
from .cove import ChainOfVerification, VerificationResult

__all__ = [
    "SelfRAG",
    "RetrievalQuality",
    "CorrectiveRAG",
    "ChainOfVerification",
    "VerificationResult",
]
