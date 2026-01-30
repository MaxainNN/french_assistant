"""
Модуль детекции галлюцинаций.

Методы проверки:
1. Lexical Grounding - лексическое заземление на контекст
2. Semantic Consistency - внутренняя согласованность ответа
3. Confidence Calibration - калибровка уверенности
"""

import re
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Детектор галлюцинаций для проверки ответов модели.

    Пример использования:
        detector = HallucinationDetector()
        result = detector.detect(response, context)
        if result["has_hallucinations"]:
            print(result["issues"])
    """

    # Маркеры уровня уверенности
    CONFIDENCE_MARKERS = {
        "high": ["всегда", "никогда", "точно", "безусловно", "100%", "гарантированно"],
        "medium": ["обычно", "как правило", "чаще всего", "в большинстве"],
        "low": ["возможно", "вероятно", "может быть", "иногда"]
    }

    def __init__(self, min_grounding_score: float = 0.3):
        """
        Инициализирует детектор галлюцинаций.

        Args:
            min_grounding_score: Минимальный порог для grounding score
        """
        self.min_grounding_score = min_grounding_score

    def check_lexical_grounding(
        self,
        response: str,
        context: str
    ) -> Tuple[float, List[str]]:
        """
        Проверяет лексическое заземление ответа на контекст.

        Args:
            response: Ответ модели
            context: Контекст из retrieved документов

        Returns:
            (grounding_score, ungrounded_terms)
        """
        # Извлекаем французские слова и технические термины
        french_terms = re.findall(r"\b[a-zéèêëàâäùûüôöîïç]{3,}\b", response.lower())
        technical_terms = re.findall(
            r"\b(?:артикль|глагол|спряжение|время|наклонение)\w*",
            response.lower()
        )

        all_terms = set(french_terms + technical_terms)
        context_lower = context.lower()

        ungrounded = []
        grounded_count = 0

        for term in all_terms:
            if term in context_lower:
                grounded_count += 1
            elif len(term) > 4:
                ungrounded.append(term)

        score = grounded_count / max(len(all_terms), 1)
        return score, ungrounded[:10]

    def check_semantic_consistency(self, response: str) -> Tuple[bool, List[str]]:
        """
        Проверяет внутреннюю согласованность ответа.

        Args:
            response: Ответ модели для проверки

        Returns:
            (is_consistent, inconsistencies)
        """
        inconsistencies = []
        sentences = re.split(r"[.!?]", response)

        # Пары противоположных утверждений
        negation_pairs = [
            ("всегда", "никогда"),
            ("можно", "нельзя"),
            ("правильно", "неправильно"),
            ("мужской род", "женский род"),
        ]

        for sent1 in sentences:
            for sent2 in sentences:
                if sent1 == sent2:
                    continue
                for pos, neg in negation_pairs:
                    if pos in sent1.lower() and neg in sent2.lower():
                        words1 = set(sent1.lower().split())
                        words2 = set(sent2.lower().split())
                        if len(words1 & words2) > 3:
                            inconsistencies.append(
                                f"Возможное противоречие: {pos} vs {neg}"
                            )

        return len(inconsistencies) == 0, inconsistencies

    def check_confidence_calibration(
        self,
        response: str
    ) -> Tuple[str, List[str]]:
        """
        Проверяет калибровку уверенности в ответе.

        Args:
            response: Ответ модели для проверки

        Returns:
            (confidence_level, overconfident_claims)
        """
        response_lower = response.lower()
        overconfident = []

        # Ищем сверхуверенные утверждения
        for marker in self.CONFIDENCE_MARKERS["high"]:
            if marker in response_lower:
                for sent in re.split(r"[.!?]", response):
                    if marker in sent.lower():
                        overconfident.append(sent.strip())

        if len(overconfident) > 2:
            return "overconfident", overconfident
        elif any(m in response_lower for m in self.CONFIDENCE_MARKERS["low"]):
            return "calibrated", []
        else:
            return "neutral", []

    def check_grounding(
        self,
        response: str,
        context: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Проверяет ответ на возможные галлюцинации (упрощённая версия).

        Args:
            response: Ответ модели
            context: Контекст из retrieved документов

        Returns:
            (is_grounded, grounding_score, ungrounded_claims)
        """
        # Извлекаем французские слова из ответа
        french_words = re.findall(
            r"\b[A-Za-zéèêëàâäùûüôöîïç]{3,}\b",
            response
        )

        context_lower = context.lower()
        grounded_count = 0
        ungrounded_claims = []

        for word in french_words:
            if word.lower() in context_lower:
                grounded_count += 1
            elif len(word) > 5:
                ungrounded_claims.append(word)

        total_words = len(french_words)
        if total_words == 0:
            return True, 1.0, []

        grounding_score = grounded_count / total_words
        is_grounded = grounding_score > self.min_grounding_score

        return is_grounded, grounding_score, ungrounded_claims[:5]

    def detect(self, response: str, context: str) -> Dict[str, Any]:
        """
        Выполняет полную детекцию галлюцинаций.

        Args:
            response: Ответ модели для проверки
            context: Контекст из retrieved документов

        Returns:
            Dict с результатами всех проверок:
            - has_hallucinations: bool
            - confidence: float
            - grounding_score: float
            - issues: List[str]
            - recommendations: List[str]
        """
        results = {
            "has_hallucinations": False,
            "confidence": 1.0,
            "issues": [],
            "recommendations": []
        }

        # 1. Лексическое заземление
        grounding_score, ungrounded = self.check_lexical_grounding(response, context)
        results["grounding_score"] = grounding_score
        results["ungrounded_terms"] = ungrounded

        if grounding_score < self.min_grounding_score:
            results["issues"].append("Низкое лексическое заземление на контекст")
            results["recommendations"].append("Добавьте ссылки на источники")
            results["has_hallucinations"] = True

        # 2. Семантическая согласованность
        is_consistent, inconsistencies = self.check_semantic_consistency(response)
        results["is_consistent"] = is_consistent
        results["inconsistencies"] = inconsistencies

        if not is_consistent:
            results["issues"].append("Обнаружены внутренние противоречия")
            results["recommendations"].append("Проверьте согласованность утверждений")
            results["has_hallucinations"] = True

        # 3. Калибровка уверенности
        conf_level, overconfident = self.check_confidence_calibration(response)
        results["confidence_level"] = conf_level
        results["overconfident_claims"] = overconfident

        if conf_level == "overconfident":
            results["issues"].append("Обнаружены сверхуверенные утверждения")
            results["recommendations"].append("Добавьте модификаторы неопределённости")

        # 4. Общая оценка уверенности
        results["confidence"] = (
            grounding_score * 0.4 +
            (1.0 if is_consistent else 0.5) * 0.3 +
            (0.7 if conf_level == "calibrated" else 1.0 if conf_level == "neutral" else 0.5) * 0.3
        )

        logger.info(
            f"Hallucination detection: has_hallucinations={results['has_hallucinations']}, "
            f"confidence={results['confidence']:.2f}"
        )

        return results
