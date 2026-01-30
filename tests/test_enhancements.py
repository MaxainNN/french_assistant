"""
Тесты RAG-улучшений.
"""

import pytest
from french_assistant.enhancements import (
    SelfRAG,
    CorrectiveRAG,
    ChainOfVerification,
    RetrievalQuality,
)
from french_assistant.retrieval import QueryExpander


class TestSelfRAG:
    """Тесты Self-RAG."""

    @pytest.fixture
    def self_rag(self):
        return SelfRAG()

    @pytest.mark.parametrize("query,expected", [
        ("Переведи: Я работаю", True),
        ("Как спрягается avoir?", True),
        ("Привет!", False),
        ("Спасибо", False),
    ])
    def test_retrieval_need(self, self_rag, query, expected):
        needs, confidence = self_rag.assess_retrieval_need(query)
        assert needs == expected
        assert 0 <= confidence <= 1

    def test_relevance_assessment(self, self_rag):
        query = "спряжение parler"
        doc = "parler (говорить): je parle, tu parles, il parle, nous parlons"

        quality, score = self_rag.assess_relevance(query, doc)
        assert quality in [RetrievalQuality.EXCELLENT, RetrievalQuality.GOOD]

    def test_utility_assessment(self, self_rag):
        query = "Как сказать привет?"
        good_response = "📝 **Перевод:** Bonjour!\n💡 Также можно: Salut! (неформально)"
        bad_response = "Ок"

        is_useful, _ = self_rag.assess_utility(query, good_response)
        assert is_useful

        is_useful, _ = self_rag.assess_utility(query, bad_response)
        assert not is_useful


class TestCorrectiveRAG:
    """Тесты CRAG."""

    @pytest.fixture
    def crag(self):
        return CorrectiveRAG()

    def test_excellent_quality_no_correction(self, crag):
        query = "спряжение parler"
        docs = ["parler: je parle, tu parles, il parle, nous parlons, vous parlez"]

        quality = crag.evaluate_retrieval_quality(query, docs)
        assert quality == RetrievalQuality.EXCELLENT

        _, meta = crag.correct(query, docs)
        assert meta["strategy"] == "none"
        assert not meta["correction_applied"]

    def test_poor_quality_uses_fallback(self, crag):
        query = "артикли во французском"
        docs = ["Погода сегодня хорошая", "Москва столица"]

        quality = crag.evaluate_retrieval_quality(query, docs)
        assert quality == RetrievalQuality.POOR

        corrected, meta = crag.correct(query, docs)
        assert meta["strategy"] == "fallback"
        assert meta["correction_applied"]
        assert any("Базовые знания" in d for d in corrected)


class TestQueryExpander:
    """Тесты QueryExpander."""

    @pytest.fixture
    def expander(self):
        return QueryExpander()

    def test_expands_with_synonyms(self, expander):
        queries = expander.expand_with_synonyms("перевод слова")
        assert len(queries) > 1
        assert "перевод слова" in queries

    def test_expand_query(self, expander):
        queries = expander.expand_query("глагол être", use_hyde=False)
        assert len(queries) >= 1

    def test_hyde_generation(self, expander):
        hyde_doc = expander.generate_hyde_document("спряжение avoir")
        assert len(hyde_doc) > 0
        assert "французск" in hyde_doc.lower() or "avoir" in hyde_doc.lower()


class TestChainOfVerification:
    """Тесты CoVe."""

    @pytest.fixture
    def cove(self):
        return ChainOfVerification()

    def test_extracts_claims(self, cove):
        response = "Перевод: Bonjour. Например: Bonjour, comment ça va?"
        claims = cove.extract_claims(response)
        assert len(claims) > 0

    def test_verifies_grounded_claim(self, cove):
        claim = "глагол parler первой группы"
        context = "parler - глагол первой группы на -er"

        is_verified, confidence, _ = cove.verify_claim(claim, context)
        assert is_verified
        assert confidence > 0.3

    def test_verification_result(self, cove):
        response = "Глагол parler спрягается: je parle, tu parles"
        context = "parler: je parle, tu parles, il parle"

        result = cove.run_verification(response, context)
        assert result.is_verified
        assert result.confidence > 0
