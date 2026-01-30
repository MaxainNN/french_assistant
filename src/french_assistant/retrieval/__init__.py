"""
Модуль retrieval для french_assistant.

Компоненты:
- QueryExpander: Расширение поисковых запросов
- VectorStoreManager: Управление векторным хранилищем
- EnhancedRetriever: Улучшенный ретривер с MMR и multi-query
"""

from .query_expansion import QueryExpander
from .vectorstore import VectorStoreManager
from .retriever import EnhancedRetriever

__all__ = [
    "QueryExpander",
    "VectorStoreManager",
    "EnhancedRetriever",
]
