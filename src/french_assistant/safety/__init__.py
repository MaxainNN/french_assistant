"""
Модуль безопасности french_assistant.

Компоненты:
- SafetyFilter: Фильтрация входных запросов
- HallucinationDetector: Детекция галлюцинаций в ответах
"""

from .filter import SafetyFilter
from .hallucination import HallucinationDetector

__all__ = [
    "SafetyFilter",
    "HallucinationDetector",
]
