"""
Главный класс FrenchAssistant.

Объединяет все компоненты RAG-системы:
- Safety Filter
- Retrieval
- Generation
- Hallucination Detection
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate

from .config import AssistantConfig, load_config, get_openai_api_key
from ..utils.tracing import TracingManager
from ..safety.filter import SafetyFilter
from ..safety.hallucination import HallucinationDetector
from ..retrieval.vectorstore import VectorStoreManager
from ..retrieval.retriever import EnhancedRetriever

logger = logging.getLogger(__name__)


class FrenchAssistant:
    """
    Главный класс нейро-сотрудника — переводчика-ассистента.

    Функционал:
    - Перевод с русского на французский
    - Объяснение грамматики
    - Работа с идиомами и фразеологизмами
    - Помощь с типичными ошибками

    Пример использования:
        assistant = FrenchAssistant()
        result = assistant.process_query("Как спрягается глагол être?")
        print(result["response"])
    """

    def __init__(
        self,
        config: AssistantConfig = None,
        config_path: Optional[str] = None
    ):
        """
        Инициализация ассистента.

        Args:
            config: Объект конфигурации (опционально)
            config_path: Путь к файлу конфигурации (опционально)
        """
        logger.info("Initializing FrenchAssistant...")

        # Загрузка конфигурации
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Инициализация компонентов
        self.tracer = TracingManager()
        self.safety_filter = SafetyFilter(self.config.safety)
        self.hallucination_detector = HallucinationDetector(
            min_grounding_score=self.config.safety.min_grounding_score
        )

        # Векторное хранилище и retriever
        self.vectorstore_manager = VectorStoreManager(
            config=self.config.vector_db,
            chunking_config=self.config.rag.chunking
        )
        self.retriever = EnhancedRetriever(
            vectorstore=self.vectorstore_manager.vectorstore,
            config=self.config.rag.retrieval,
            tracer=self.tracer
        )

        # Инициализация LLM
        self.llm = self._init_llm()

        # Промпт
        self.prompt_template = self._create_prompt_template()

        logger.info("FrenchAssistant initialized successfully!")

    def _init_llm(self):
        """
        Инициализирует LLM модель.

        Поддерживает OpenAI (по умолчанию).
        Возвращает None если API ключ не настроен.
        """
        if self.config.model.provider == "openai":
            api_key = get_openai_api_key()
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY не найден в переменных окружения. "
                    "LLM будет работать в демо-режиме с шаблонными ответами."
                )
                return None

            try:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=self.config.model.model_name,
                    temperature=self.config.model.temperature,
                    max_tokens=self.config.model.max_new_tokens,
                    api_key=api_key,
                )
                logger.info(f"OpenAI LLM инициализирован: {self.config.model.model_name}")
                return llm

            except ImportError:
                logger.error(
                    "langchain-openai не установлен. "
                    "Установите: pip install langchain-openai"
                )
                return None
            except Exception as e:
                logger.error(f"Ошибка инициализации OpenAI LLM: {e}")
                return None

        logger.warning(f"Неизвестный провайдер: {self.config.model.provider}")
        return None

    def _create_prompt_template(self) -> PromptTemplate:
        """Создаёт шаблон промпта."""
        template = """
            {system_prompt}

            ## Контекст из базы знаний:
            {context}

            ## Вопрос пользователя:
            {question}

            ## Твой ответ:
        """
        return PromptTemplate(
            input_variables=["system_prompt", "context", "question"],
            template=template
        )

    def _generate_response(self, query: str, context: str) -> str:
        """
        Генерирует ответ на основе контекста.

        Args:
            query: Запрос пользователя
            context: Контекст из retrieved документов

        Returns:
            Сгенерированный ответ
        """
        if self.llm is None:
            return self._generate_template_response(query, context)

        prompt = self.prompt_template.format(
            system_prompt=self.config.system_prompt,
            context=context,
            question=query
        )

        response = self.llm.invoke(prompt)
        return response.content

    def _generate_template_response(self, query: str, context: str) -> str:
        """Генерирует шаблонный ответ (для демонстрации без LLM)."""
        query_lower = query.lower()

        if "перевед" in query_lower or "перевод" in query_lower:
            return self._format_translation_response(query, context)
        elif "как сказать" in query_lower:
            return self._format_how_to_say_response(query, context)
        elif "грамматик" in query_lower or "правил" in query_lower:
            return self._format_grammar_response(query, context)
        elif "идиом" in query_lower or "выражен" in query_lower:
            return self._format_idiom_response(query, context)
        else:
            return self._format_general_response(query, context)

    def _format_translation_response(self, query: str, context: str) -> str:
        """Форматирует ответ на запрос перевода."""
        text_match = re.search(
            r'перевед[и|ите]?\s*:?\s*["\']?(.+?)["\']?\s*$',
            query,
            re.IGNORECASE
        )
        if text_match:
            text_to_translate = text_match.group(1)
        else:
            text_to_translate = query.replace("переведи", "").replace("перевод", "").strip()

        return f"""📝 **Запрос на перевод:** "{text_to_translate}"

            💡 **Комментарий:** Для точного перевода использована информация из базы знаний.

            📚 **Релевантный контекст из базы:**
            {context[:500]}...

            ⚠️ **Примечание:** Для качественного перевода укажите контекст использования."""

    def _format_how_to_say_response(self, query: str, context: str) -> str:
        """Форматирует ответ на вопрос 'как сказать'."""
        return f"""📝 **Ваш вопрос:** {query}

            📚 **Информация из базы знаний:**
            {context[:600]}

            💡 **Совет:** Обратите внимание на контекст и регистр речи."""

    def _format_grammar_response(self, query: str, context: str) -> str:
        """Форматирует ответ на грамматический вопрос."""
        return f"""📖 **Грамматический вопрос:** {query}

        📚 **Информация из базы знаний:**
        {context[:700]}

        💡 **Комментарий:** Обратите внимание на примеры использования."""

    def _format_idiom_response(self, query: str, context: str) -> str:
        """Форматирует ответ об идиомах."""
        return f"""🗣️ **Вопрос об идиомах:** {query}

        📚 **Информация из базы знаний:**
        {context[:700]}

        💡 **Совет:** Идиомы часто не переводятся буквально."""

    def _format_general_response(self, query: str, context: str) -> str:
        """Форматирует общий ответ."""
        return f"""📝 **Ваш вопрос:** {query}

        📚 **Релевантная информация из базы знаний:**
        {context[:700]}

        💡 **Примечание:** Уточните вопрос для более детального ответа."""

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Обрабатывает запрос пользователя.

        Полный pipeline:
        1. Safety check
        2. Retrieval
        3. Response generation
        4. Hallucination check

        Args:
            query: Запрос пользователя

        Returns:
            Dict с ответом и метаданными:
            - query: исходный запрос
            - response: сгенерированный ответ
            - sources: список источников
            - metadata: метаданные обработки
            - is_safe: прошёл ли safety check
            - error: сообщение об ошибке (если есть)
        """
        start_time = time.time()

        result = {
            "query": query,
            "response": "",
            "sources": [],
            "metadata": {},
            "trace": None,
            "is_safe": True,
            "error": None
        }

        self.tracer.log_event("query_received", "FrenchAssistant", query, "processing")

        # 1. Safety check
        is_safe, error_msg, safety_meta = self.safety_filter.filter_input(query)
        result["metadata"]["safety"] = safety_meta

        if not is_safe:
            result["is_safe"] = False
            result["response"] = error_msg
            result["error"] = error_msg
            self.tracer.log_event("safety_blocked", "SafetyFilter", query, error_msg)
            return result

        self.tracer.log_event("safety_passed", "SafetyFilter", query, "safe")

        # 2. Retrieval
        try:
            docs = self.retriever.retrieve(query)
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])

            self.tracer.log_event(
                "retrieval_complete",
                "EnhancedRetriever",
                query,
                f"{len(docs)} documents retrieved"
            )

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            result["error"] = f"Ошибка поиска: {str(e)}"
            result["response"] = "Извините, произошла ошибка при поиске информации."
            return result

        # 3. Response generation
        try:
            response = self._generate_response(query, context)
            result["response"] = response

            self.tracer.log_event(
                "response_generated",
                "LLM",
                query[:50] + "...",
                response[:100] + "..."
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            result["error"] = f"Ошибка генерации: {str(e)}"
            result["response"] = "Извините, произошла ошибка при генерации ответа."
            return result

        # 4. Hallucination check
        hallucination_result = self.hallucination_detector.detect(response, context)
        result["metadata"]["grounding"] = {
            "is_grounded": not hallucination_result["has_hallucinations"],
            "score": hallucination_result["grounding_score"],
            "confidence": hallucination_result["confidence"]
        }

        if hallucination_result["has_hallucinations"]:
            logger.warning("Potential hallucination detected")
            result["response"] += "\n\n⚠️ *Некоторые части ответа могут требовать проверки.*"

        # Финализация
        duration = (time.time() - start_time) * 1000
        result["metadata"]["total_duration_ms"] = duration
        result["trace"] = self.tracer.get_trace_report()

        self.tracer.log_event(
            "query_complete",
            "FrenchAssistant",
            query[:50] + "...",
            f"Response generated in {duration:.2f}ms"
        )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику работы ассистента."""
        return {
            "total_events": len(self.tracer.events),
            "collection_count": self.vectorstore_manager.get_collection_count(),
            "config": {
                "embedding_model": self.config.vector_db.embeddings.model,
                "chunk_size": self.config.rag.chunking.chunk_size,
                "retrieval_k": self.config.rag.retrieval.k
            }
        }
