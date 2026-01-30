"""
Модуль Chain-of-Verification (CoVe).

Процесс верификации:
1. Извлечение утверждений (claims) из ответа
2. Верификация каждого claim против контекста
3. Сбор evidence для подтверждения
4. Итоговая оценка достоверности
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Результат верификации ответа."""
    is_verified: bool
    confidence: float
    issues: List[str]
    corrections: List[str]
    grounding_evidence: List[str]


class ChainOfVerification:
    """
    Chain-of-Verification (CoVe) для проверки ответов.

    Процесс:
    1. Извлечение claims из ответа
    2. Верификация каждого claim против контекста
    3. Агрегация результатов
    4. Генерация отчёта о верификации

    Пример использования:
        cove = ChainOfVerification()
        result = cove.run_verification(response, context)
        if not result.is_verified:
            print(result.issues)
    """

    # Шаблоны для генерации проверочных вопросов
    VERIFICATION_TEMPLATES = {
        "factual": "Является ли утверждение '{claim}' фактически верным?",
        "consistency": "Согласуется ли '{claim}' с другими частями ответа?",
        "completeness": "Полностью ли раскрыт вопрос '{aspect}'?",
        "source": "Есть ли в базе знаний подтверждение для '{claim}'?"
    }

    def __init__(self, min_confidence: float = 0.5):
        """
        Инициализирует CoVe.

        Args:
            min_confidence: Минимальный порог уверенности для верификации
        """
        self.min_confidence = min_confidence

    def extract_claims(self, response: str) -> List[str]:
        """
        Извлекает ключевые утверждения из ответа.

        Args:
            response: Текст ответа

        Returns:
            Список утверждений (claims)
        """
        claims = []

        # 1. Предложения с переводами
        translation_pattern = r"(?:перевод|французски|traduire)[:\s]+([^.!?\n]+)"
        claims.extend(re.findall(translation_pattern, response, re.IGNORECASE))

        # 2. Грамматические утверждения
        grammar_pattern = r"(?:правило|используется|образуется|спрягается)[:\s]+([^.!?\n]+)"
        claims.extend(re.findall(grammar_pattern, response, re.IGNORECASE))

        # 3. Примеры
        example_pattern = r"(?:например|пример)[:\s]+([^.!?\n]+)"
        claims.extend(re.findall(example_pattern, response, re.IGNORECASE))

        return list(set(claims))[:10]

    def verify_claim(
        self,
        claim: str,
        context: str
    ) -> Tuple[bool, float, str]:
        """
        Верифицирует отдельное утверждение.

        Args:
            claim: Утверждение для проверки
            context: Контекст для верификации

        Returns:
            (is_verified, confidence, evidence)
        """
        claim_lower = claim.lower()
        context_lower = context.lower()

        # Извлекаем ключевые слова
        claim_words = set(re.findall(r"\b\w{4,}\b", claim_lower))
        context_words = set(re.findall(r"\b\w{4,}\b", context_lower))

        # Пересечение
        overlap = claim_words & context_words

        if len(claim_words) == 0:
            return True, 1.0, "Empty claim"

        overlap_ratio = len(overlap) / len(claim_words)

        # Ищем прямое подтверждение в контексте
        evidence = ""
        for word in overlap:
            sentences = re.split(r"[.!?]", context)
            for sent in sentences:
                if word in sent.lower():
                    evidence = sent.strip()
                    break

        is_verified = overlap_ratio > 0.3
        confidence = min(overlap_ratio * 1.5, 1.0)

        return is_verified, confidence, evidence

    def run_verification(
        self,
        response: str,
        context: str
    ) -> VerificationResult:
        """
        Выполняет полную верификацию ответа.

        Args:
            response: Ответ для верификации
            context: Контекст (retrieved документы)

        Returns:
            VerificationResult с результатами проверки
        """
        claims = self.extract_claims(response)

        issues = []
        corrections = []
        grounding_evidence = []

        verified_count = 0
        total_confidence = 0.0

        for claim in claims:
            is_verified, confidence, evidence = self.verify_claim(claim, context)
            total_confidence += confidence

            if is_verified:
                verified_count += 1
                if evidence:
                    grounding_evidence.append(evidence[:100])
            else:
                issues.append(f"Не подтверждено: {claim[:50]}...")
                corrections.append(f"Рекомендуется проверить: {claim[:50]}...")

        overall_verified = verified_count / max(len(claims), 1) > self.min_confidence
        avg_confidence = total_confidence / max(len(claims), 1)

        return VerificationResult(
            is_verified=overall_verified,
            confidence=avg_confidence,
            issues=issues,
            corrections=corrections,
            grounding_evidence=grounding_evidence[:5]
        )
