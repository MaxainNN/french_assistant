"""
Модуль CRAG - Corrective Retrieval Augmented Generation.

Стратегия:
1. Оценка качества retrieved документов
2. Выбор стратегии коррекции
3. Применение коррекции (fallback, supplement, refine)
"""

import logging
from typing import Any, Dict, List, Tuple

from .self_rag import SelfRAG, RetrievalQuality

logger = logging.getLogger(__name__)


class CorrectiveRAG:
    """
    CRAG - Corrective Retrieval Augmented Generation.

    Оценивает качество retrieved документов и применяет
    корректирующие стратегии при необходимости.

    Стратегии:
    - none: коррекция не нужна (EXCELLENT quality)
    - supplement: дополнить базовыми знаниями (GOOD quality)
    - refine: уточнить поиск (PARTIAL quality)
    - fallback: использовать fallback-знания (POOR quality)

    Пример использования:
        crag = CorrectiveRAG()
        corrected_docs, metadata = crag.correct(query, documents)
    """

    # Fallback знания (базовые факты о французском)
    DEFAULT_FALLBACK_KNOWLEDGE = {
        "articles": (
            "Во французском языке 3 типа артиклей: определённые (le, la, les), "
            "неопределённые (un, une, des) и частичные (du, de la, de l')."
        ),
        "verbs": (
            "Французские глаголы делятся на 3 группы: -er (1-я), -ir с -issons (2-я) "
            "и неправильные (3-я). Aller - исключение из 1-й группы."
        ),
        "tenses": (
            "Основные времена: présent (настоящее), passé composé (прошедшее составное), "
            "imparfait (незавершенное), futur simple (простое будущее)."
        ),
    }

    def __init__(self, fallback_knowledge: dict = None):
        """
        Инициализирует CRAG.

        Args:
            fallback_knowledge: Словарь fallback-знаний (опционально)
        """
        self.self_rag = SelfRAG()
        self.fallback_knowledge = fallback_knowledge or self.DEFAULT_FALLBACK_KNOWLEDGE

    def evaluate_retrieval_quality(
        self,
        query: str,
        documents: List[str]
    ) -> RetrievalQuality:
        """
        Оценивает общее качество retrieved документов.

        Args:
            query: Запрос пользователя
            documents: Список retrieved документов

        Returns:
            Оценка качества (RetrievalQuality)
        """
        if not documents:
            return RetrievalQuality.POOR

        qualities = []
        for doc in documents:
            quality, _ = self.self_rag.assess_relevance(query, doc)
            qualities.append(quality)

        # Берём лучшее качество из топ-3
        quality_order = [
            RetrievalQuality.EXCELLENT,
            RetrievalQuality.GOOD,
            RetrievalQuality.PARTIAL,
            RetrievalQuality.POOR
        ]

        for q in quality_order:
            if q in qualities[:3]:
                return q

        return RetrievalQuality.POOR

    def get_correction_strategy(self, quality: RetrievalQuality) -> str:
        """
        Определяет стратегию коррекции на основе качества.

        Args:
            quality: Оценка качества документов

        Returns:
            Название стратегии коррекции
        """
        strategies = {
            RetrievalQuality.EXCELLENT: "none",
            RetrievalQuality.GOOD: "supplement",
            RetrievalQuality.PARTIAL: "refine",
            RetrievalQuality.POOR: "fallback",
            RetrievalQuality.AMBIGUOUS: "clarify"
        }
        return strategies.get(quality, "fallback")

    def apply_correction(
        self,
        query: str,
        documents: List[str],
        strategy: str
    ) -> Tuple[List[str], str]:
        """
        Применяет стратегию коррекции.

        Args:
            query: Запрос пользователя
            documents: Исходные документы
            strategy: Стратегия коррекции

        Returns:
            (corrected_documents, correction_note)
        """
        if strategy == "none":
            return documents, ""

        elif strategy == "supplement":
            # Добавляем релевантные fallback знания
            note = "Добавлена базовая информация для полноты ответа."
            query_lower = query.lower()

            for topic, knowledge in self.fallback_knowledge.items():
                if topic in query_lower or any(
                    kw in query_lower for kw in topic.split()
                ):
                    documents.append(f"[Базовые знания] {knowledge}")
                    break

            return documents, note

        elif strategy == "refine":
            note = "Качество контекста ограничено. Рекомендуется уточнить вопрос."
            return documents, note

        elif strategy == "fallback":
            note = "Релевантная информация не найдена. Используются базовые знания."
            fallback_docs = [
                f"[Базовые знания] {k}"
                for k in self.fallback_knowledge.values()
            ]
            return fallback_docs, note

        elif strategy == "clarify":
            note = "Найдена противоречивая информация. Требуется уточнение вопроса."
            return documents, note

        return documents, ""

    def correct(
        self,
        query: str,
        documents: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Основной метод коррекции.

        Args:
            query: Запрос пользователя
            documents: Retrieved документы

        Returns:
            (corrected_documents, correction_metadata)

        Пример:
            crag = CorrectiveRAG()
            docs, meta = crag.correct("спряжение avoir", retrieved_docs)
            # meta = {"strategy": "supplement", "correction_applied": True, ...}
        """
        # 1. Оценка качества
        quality = self.evaluate_retrieval_quality(query, documents)

        # 2. Определение стратегии
        strategy = self.get_correction_strategy(quality)

        # 3. Применение коррекции
        corrected_docs, note = self.apply_correction(query, documents, strategy)

        metadata = {
            "original_quality": quality.value,
            "strategy": strategy,
            "correction_applied": strategy != "none",
            "note": note,
            "original_doc_count": len(documents),
            "corrected_doc_count": len(corrected_docs)
        }

        logger.info(f"CRAG correction: {quality.value} -> {strategy}")

        return corrected_docs, metadata
