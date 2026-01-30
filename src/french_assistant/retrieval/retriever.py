"""
Модуль улучшенного ретривера.

Техники:
1. Multi-query retrieval
2. MMR (Maximum Marginal Relevance)
3. Re-ranking
"""

import hashlib
import logging
import time
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from ..core.config import RetrievalConfig
from ..utils.tracing import TracingManager
from .query_expansion import QueryExpander

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """
    Улучшенный ретривер с несколькими техниками поиска.

    Техники:
    1. Multi-query retrieval - поиск по нескольким вариантам запроса
    2. MMR - баланс между релевантностью и разнообразием
    3. Simple re-ranking - переранжирование по совпадению терминов

    Пример использования:
        retriever = EnhancedRetriever(vectorstore, config, tracer)
        docs = retriever.retrieve("спряжение глагола être")
    """

    def __init__(
        self,
        vectorstore: Chroma,
        config: RetrievalConfig = None,
        tracer: TracingManager = None
    ):
        """
        Инициализирует улучшенный ретривер.

        Args:
            vectorstore: Векторное хранилище ChromaDB
            config: Конфигурация поиска
            tracer: Менеджер трассировки (опционально)
        """
        self.vectorstore = vectorstore
        self.config = config or RetrievalConfig()
        self.tracer = tracer

        # Базовый retriever с MMR
        self.base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.k,
                "fetch_k": self.config.fetch_k,
                "lambda_mult": self.config.lambda_mult
            }
        )

        # Query expander
        self.query_expander = QueryExpander()

        logger.info("EnhancedRetriever initialized with MMR search")

    def _log_event(
        self,
        event_type: str,
        input_data,
        output_data,
        metadata: dict = None,
        duration_ms: float = 0.0
    ) -> None:
        """Логирует событие, если трассировка включена."""
        if self.tracer:
            self.tracer.log_event(
                event_type,
                "EnhancedRetriever",
                input_data,
                output_data,
                metadata,
                duration_ms
            )

    def retrieve(
        self,
        query: str,
        use_expansion: bool = True,
        use_hyde: bool = True
    ) -> List[Document]:
        """
        Выполняет улучшенный поиск документов.

        Args:
            query: Поисковый запрос
            use_expansion: Использовать расширение запроса
            use_hyde: Использовать HyDE

        Returns:
            Список релевантных документов
        """
        start_time = time.time()

        # 1. Расширяем запрос
        if use_expansion:
            expanded_queries = self.query_expander.expand_query(
                query,
                use_hyde=use_hyde
            )
        else:
            expanded_queries = [query]

        self._log_event(
            "query_expansion",
            query,
            f"{len(expanded_queries)} variants",
            {"variants": expanded_queries}
        )

        # 2. Поиск по всем вариантам запроса
        all_docs = []
        seen_contents = set()

        for q in expanded_queries:
            docs = self.base_retriever.get_relevant_documents(q)
            for doc in docs:
                # Дедупликация по содержимому
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)

        self._log_event(
            "multi_query_retrieval",
            f"{len(expanded_queries)} queries",
            f"{len(all_docs)} unique docs"
        )

        # 3. Ранжирование по релевантности
        scored_docs = self._rank_documents(query, all_docs)

        # 4. Возвращаем топ-k
        top_k = self.config.top_k_final
        result_docs = [doc for _, doc in scored_docs[:top_k]]

        duration = (time.time() - start_time) * 1000
        self._log_event(
            "retrieval_complete",
            query,
            f"{len(result_docs)} final docs",
            {"scores": [s for s, _ in scored_docs[:top_k]]},
            duration
        )

        return result_docs

    def _rank_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[tuple]:
        """
        Ранжирует документы по релевантности.

        Использует простую эвристику на основе совпадения терминов.

        Args:
            query: Исходный запрос
            documents: Список документов для ранжирования

        Returns:
            Список кортежей (score, document), отсортированный по убыванию score
        """
        scored_docs = []
        query_terms = set(query.lower().split())

        for doc in documents:
            content_lower = doc.page_content.lower()

            # Подсчёт совпадений с терминами запроса
            term_score = sum(1 for term in query_terms if term in content_lower)

            # Бонус за длину (но не слишком длинные)
            length_score = min(len(doc.page_content) / 500, 1.0)

            final_score = term_score + length_score * 0.5
            scored_docs.append((final_score, doc))

        # Сортировка по убыванию score
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return scored_docs

    def similarity_search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """
        Простой поиск по сходству без расширения запроса.

        Args:
            query: Поисковый запрос
            k: Количество результатов

        Returns:
            Список документов
        """
        k = k or self.config.k
        return self.vectorstore.similarity_search(query, k=k)
