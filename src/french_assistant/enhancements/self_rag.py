"""
Модуль Self-RAG: самооценка качества retrieval и генерации.

Компоненты:
1. Retrieval Token - оценка необходимости retrieval
2. Relevance Token - оценка релевантности документа
3. Support Token - оценка поддержки ответа документами
4. Utility Token - оценка полезности ответа
"""

import re
import logging
from enum import Enum
from typing import List, Tuple

logger = logging.getLogger(__name__)


class RetrievalQuality(Enum):
    """Оценка качества retrieved документов."""
    EXCELLENT = "excellent"  # Прямой ответ на вопрос
    GOOD = "good"           # Релевантная информация
    PARTIAL = "partial"     # Частично релевантная
    POOR = "poor"           # Нерелевантная
    AMBIGUOUS = "ambiguous" # Противоречивая информация


class SelfRAG:
    """
    Self-RAG: самооценка качества retrieval и генерации.

    Реализует токены самооценки для определения:
    - Нужен ли retrieval для данного запроса
    - Насколько релевантны найденные документы
    - Поддерживается ли ответ документами
    - Насколько полезен ответ

    Пример использования:
        self_rag = SelfRAG()
        needs_retrieval, conf = self_rag.assess_retrieval_need(query)
        quality, score = self_rag.assess_relevance(query, document)
    """

    # Ключевые слова-триггеры для определения необходимости retrieval
    DEFAULT_RETRIEVAL_TRIGGERS = {
        "high": ["перевод", "как сказать", "правило", "грамматика", "спряжение"],
        "low": ["привет", "спасибо", "пока", "как дела"]
    }

    def __init__(self, retrieval_triggers: dict = None):
        """
        Инициализирует Self-RAG.

        Args:
            retrieval_triggers: Словарь триггеров для retrieval
        """
        self.retrieval_triggers = retrieval_triggers or self.DEFAULT_RETRIEVAL_TRIGGERS

    def assess_retrieval_need(self, query: str) -> Tuple[bool, float]:
        """
        Оценивает, нужен ли retrieval для данного запроса.

        Args:
            query: Запрос пользователя

        Returns:
            (needs_retrieval, confidence)
        """
        query_lower = query.lower()

        # Подсчёт триггеров
        high_triggers = sum(
            1 for t in self.retrieval_triggers["high"]
            if t in query_lower
        )
        low_triggers = sum(
            1 for t in self.retrieval_triggers["low"]
            if t in query_lower
        )

        if low_triggers > high_triggers:
            return False, 0.9
        elif high_triggers > 0:
            return True, min(0.6 + high_triggers * 0.1, 1.0)
        else:
            # По умолчанию retrieval нужен
            return True, 0.7

    def assess_relevance(
        self,
        query: str,
        document: str
    ) -> Tuple[RetrievalQuality, float]:
        """
        Оценивает релевантность документа запросу.

        Args:
            query: Запрос пользователя
            document: Текст документа

        Returns:
            (quality, confidence_score)
        """
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        # Jaccard similarity
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        jaccard = intersection / max(union, 1)

        # Проверяем наличие ключевых терминов запроса
        key_terms_found = sum(
            1 for w in query_words
            if len(w) > 4 and w in document.lower()
        )

        # Оценка качества
        if jaccard > 0.3 and key_terms_found >= 2:
            return RetrievalQuality.EXCELLENT, 0.9
        elif jaccard > 0.2 or key_terms_found >= 1:
            return RetrievalQuality.GOOD, 0.7
        elif jaccard > 0.1:
            return RetrievalQuality.PARTIAL, 0.5
        else:
            return RetrievalQuality.POOR, 0.3

    def assess_support(
        self,
        response: str,
        documents: List[str]
    ) -> Tuple[bool, float]:
        """
        Оценивает, поддерживается ли ответ документами.

        Args:
            response: Сгенерированный ответ
            documents: Список документов-источников

        Returns:
            (is_supported, support_ratio)
        """
        combined_docs = " ".join(documents).lower()
        response_lower = response.lower()

        # Извлекаем существенные слова из ответа
        response_words = set(
            re.findall(r"\b[a-zа-яéèêëàâäùûüôöîïç]{4,}\b", response_lower)
        )

        # Считаем, сколько из них есть в документах
        supported = sum(1 for w in response_words if w in combined_docs)

        support_ratio = supported / max(len(response_words), 1)

        return support_ratio > 0.3, support_ratio

    def assess_utility(self, query: str, response: str) -> Tuple[bool, float]:
        """
        Оценивает полезность ответа для запроса.

        Args:
            query: Запрос пользователя
            response: Сгенерированный ответ

        Returns:
            (is_useful, utility_score)
        """
        # Проверяем, что ответ не пустой и содержательный
        if len(response) < 50:
            return False, 0.2

        # Проверяем наличие структуры ответа
        has_structure = any(
            marker in response
            for marker in ["📝", "💡", "📚", "⚠️", "**"]
        )

        # Проверяем, что ответ затрагивает тему запроса
        query_topics = set(re.findall(r"\b\w{4,}\b", query.lower()))
        response_topics = set(re.findall(r"\b\w{4,}\b", response.lower()))

        topic_coverage = len(query_topics & response_topics) / max(len(query_topics), 1)

        utility_score = (0.3 if has_structure else 0) + topic_coverage * 0.7

        return utility_score > 0.4, utility_score
