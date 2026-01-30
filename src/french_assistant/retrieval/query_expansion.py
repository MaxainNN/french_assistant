"""
Модуль расширения запросов для улучшения поиска.

Техники:
1. Synonym expansion - расширение синонимами
2. HyDE - Hypothetical Document Embeddings
3. Multi-query generation - генерация нескольких вариантов запроса
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Расширитель запросов для улучшения качества поиска.

    Пример использования:
        expander = QueryExpander()
        queries = expander.expand_query("спряжение avoir", use_hyde=True)
        # ["спряжение avoir", "спрягается avoir", "conjugation avoir", ...]
    """

    # Словарь синонимов для ключевых терминов
    DEFAULT_SYNONYMS = {
        "перевод": ["перевести", "translation", "traduire"],
        "глагол": ["verb", "verbe", "спряжение"],
        "артикль": ["article", "определённый", "неопределённый"],
        "время": ["tense", "temps", "présent", "passé", "futur"],
        "идиома": ["выражение", "фразеологизм", "idiome", "expression"],
    }

    def __init__(self, llm=None, synonyms: dict = None):
        """
        Инициализирует расширитель запросов.

        Args:
            llm: LLM для генерации HyDE документов (опционально)
            synonyms: Словарь синонимов для расширения (опционально)
        """
        self.llm = llm
        self.synonyms = synonyms or self.DEFAULT_SYNONYMS

    def expand_with_synonyms(self, query: str) -> List[str]:
        """
        Расширяет запрос синонимами.

        Args:
            query: Исходный запрос

        Returns:
            Список вариантов запроса с синонимами
        """
        expanded = [query]
        query_lower = query.lower()

        for term, syns in self.synonyms.items():
            if term in query_lower:
                for syn in syns:
                    expanded.append(query_lower.replace(term, syn))

        return list(set(expanded))[:4]

    def generate_hyde_document(self, query: str) -> str:
        """
        Генерирует гипотетический документ для HyDE.

        HyDE (Hypothetical Document Embeddings) - техника, при которой
        генерируется гипотетический документ, отвечающий на запрос,
        и его эмбеддинг используется для поиска реальных документов.

        Args:
            query: Запрос пользователя

        Returns:
            Гипотетический документ
        """
        if self.llm:
            prompt = f"""Напиши краткий информативный абзац, который бы отвечал на вопрос:
            "{query}"
            Ответ должен быть на русском языке и касаться французской грамматики или перевода."""
            try:
                return self.llm.predict(prompt)
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}")

        # Fallback: шаблонный документ
        return (
            f"Информация о французском языке по теме: {query}. "
            f"Грамматические правила и примеры использования."
        )

    def expand_query(
        self,
        query: str,
        use_hyde: bool = False,
        max_variants: int = 4
    ) -> List[str]:
        """
        Комплексное расширение запроса.

        Args:
            query: Исходный запрос
            use_hyde: Использовать ли HyDE для генерации гипотетического документа
            max_variants: Максимальное количество вариантов

        Returns:
            Список расширенных вариантов запроса
        """
        queries = self.expand_with_synonyms(query)

        if use_hyde:
            hyde_doc = self.generate_hyde_document(query)
            queries.append(hyde_doc)

        result = queries[:max_variants]
        logger.debug(f"Query expanded: {query} -> {len(result)} variants")

        return result
