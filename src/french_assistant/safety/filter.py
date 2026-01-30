"""
Модуль фильтрации входных данных для безопасности.

Защита от:
1. Prompt injection
2. Off-topic запросов
3. Слишком длинных запросов
"""

import re
import logging
from typing import Dict, List, Tuple

from ..core.config import SafetyConfig

logger = logging.getLogger(__name__)


class SafetyFilter:
    """
    Фильтр безопасности для входных данных.

    Пример использования:
        filter = SafetyFilter(config)
        is_safe, error_msg, metadata = filter.filter_input(user_query)
        if not is_safe:
            return error_msg
    """

    # Паттерны для детекции prompt injection
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
        r"(?i)forget\s+(everything|all|what)",
        r"(?i)you\s+are\s+now\s+a",
        r"(?i)new\s+instructions?:",
        r"(?i)system\s*prompt",
        r"(?i)jailbreak",
        r"(?i)bypass\s+(the\s+)?(filter|safety|security)",
        r"(?i)act\s+as\s+(if|though)",
        r"(?i)pretend\s+(to\s+be|you\s+are)",
        r"(?i)disregard\s+(your|the)\s+(rules|instructions)",
        r"(?i)override\s+(the\s+)?(system|safety)",
    ]

    # Ключевые слова для проверки релевантности теме
    TOPIC_KEYWORDS_RU = [
        "перевод", "переведи", "французск", "франция", "как сказать",
        "по-французски", "грамматик", "артикль", "глагол", "слово",
        "выражение", "фраза", "идиом", "означает", "перевести",
        "язык", "произношение", "accent", "время глагол"
    ]

    TOPIC_KEYWORDS_FR = [
        "traduire", "traduction", "français", "russe", "grammaire",
        "verbe", "article", "expression", "mot", "phrase"
    ]

    def __init__(self, config: SafetyConfig = None):
        """
        Инициализирует фильтр безопасности.

        Args:
            config: Конфигурация безопасности (опционально)
        """
        self.config = config or SafetyConfig()
        self.max_length = self.config.max_length
        self.blocked_patterns = [re.compile(p) for p in self.INJECTION_PATTERNS]

        logger.info(
            "SafetyFilter initialized with %d injection patterns",
            len(self.INJECTION_PATTERNS)
        )

    def check_injection(self, text: str) -> Tuple[bool, str]:
        """
        Проверяет текст на prompt injection.

        Args:
            text: Входной текст для проверки

        Returns:
            (is_safe, error_message)
        """
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return False, f"Обнаружена попытка injection: {pattern.pattern}"
        return True, ""

    def check_topic_relevance(self, text: str) -> Tuple[bool, float]:
        """
        Проверяет релевантность запроса теме французского языка.

        Args:
            text: Входной текст для проверки

        Returns:
            (is_relevant, confidence_score)
        """
        text_lower = text.lower()

        # Подсчёт совпадений с ключевыми словами
        ru_matches = sum(1 for kw in self.TOPIC_KEYWORDS_RU if kw in text_lower)
        fr_matches = sum(1 for kw in self.TOPIC_KEYWORDS_FR if kw in text_lower)

        word_count = len(text.split())
        score = (ru_matches + fr_matches) / min(10, max(word_count, 1))

        # Проверка наличия французских символов
        has_french_chars = bool(re.search(r"[éèêëàâäùûüôöîïç]", text_lower))

        is_relevant = ru_matches > 0 or fr_matches > 0 or has_french_chars or score > 0.1

        return is_relevant, min(score, 1.0)

    def check_length(self, text: str) -> Tuple[bool, str]:
        """
        Проверяет длину входного текста.

        Args:
            text: Входной текст для проверки

        Returns:
            (is_ok, error_message)
        """
        if len(text) > self.max_length:
            return False, f"Текст слишком длинный ({len(text)} > {self.max_length})"
        return True, ""

    def filter_input(self, text: str) -> Tuple[bool, str, Dict]:
        """
        Комплексная фильтрация входного запроса.

        Args:
            text: Входной текст для проверки

        Returns:
            (is_safe, error_message, metadata)

        Пример:
            is_safe, error, meta = filter.filter_input("Переведи: Привет")
            # is_safe=True, error="", meta={"checks_passed": [...]}
        """
        metadata = {
            "original_length": len(text),
            "checks_passed": []
        }

        # 1. Проверка длины
        is_ok, msg = self.check_length(text)
        if not is_ok:
            return False, msg, metadata
        metadata["checks_passed"].append("length")

        # 2. Проверка на injection
        is_ok, msg = self.check_injection(text)
        if not is_ok:
            logger.warning(f"Injection detected: {text[:100]}...")
            return False, "Извините, я не могу обработать этот запрос.", metadata
        metadata["checks_passed"].append("injection")

        # 3. Проверка релевантности теме
        is_relevant, score = self.check_topic_relevance(text)
        metadata["topic_score"] = score

        if not is_relevant:
            return (
                False,
                "Извините, я специализируюсь только на переводе "
                "с русского на французский и вопросах французского языка. "
                "Пожалуйста, задайте вопрос по этой теме.",
                metadata
            )
        metadata["checks_passed"].append("topic_relevance")

        return True, "", metadata
