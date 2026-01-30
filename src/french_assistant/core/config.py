"""
Модуль загрузки и управления конфигурацией.

Предоставляет:
- AssistantConfig: датакласс с типизированной конфигурацией
- load_config: функция загрузки конфигурации из YAML
- get_hf_token: функция получения HuggingFace токена
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()


def get_hf_token() -> Optional[str]:
    """
    Получает HuggingFace токен из переменных окружения.

    Проверяет в порядке приоритета:
    1. HF_TOKEN
    2. HUGGINGFACE_TOKEN
    3. HUGGING_FACE_HUB_TOKEN

    Returns:
        Токен или None, если не найден
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return token


def get_openai_api_key() -> Optional[str]:
    """Получает OpenAI API ключ из переменных окружения."""
    return os.getenv("OPENAI_API_KEY")


@dataclass
class ModelConfig:
    """Конфигурация LLM модели."""
    provider: str = "openai"  # "openai" или "huggingface"
    model_name: str = "gpt-4o-mini"  # Модель OpenAI
    primary_model: str = "IlyaGusev/saiga_llama3_8b"  # Для HuggingFace (legacy)
    alternative_models: List[str] = field(default_factory=list)
    temperature: float = 0.3
    max_new_tokens: int = 1024
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class EmbeddingsConfig:
    """Конфигурация эмбеддингов."""
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"


@dataclass
class VectorDBConfig:
    """Конфигурация векторной базы данных."""
    type: str = "chromadb"
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "french_knowledge"
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)


@dataclass
class ChunkingConfig:
    """Конфигурация разбиения документов."""
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ".", "!", "?", ";", ","]
    )


@dataclass
class RetrievalConfig:
    """Конфигурация поиска."""
    search_type: str = "mmr"
    k: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.7
    top_k_final: int = 3


@dataclass
class RAGConfig:
    """Конфигурация RAG pipeline."""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


@dataclass
class SafetyConfig:
    """Конфигурация безопасности."""
    max_length: int = 2000
    min_grounding_score: float = 0.6
    enabled: bool = True


@dataclass
class TracingConfig:
    """Конфигурация трассировки."""
    enabled: bool = True
    log_level: str = "DEBUG"
    log_file: str = "./logs/french_assistant.log"


@dataclass
class AssistantConfig:
    """
    Главный класс конфигурации French Assistant.

    Объединяет все конфигурации в единую структуру.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    system_prompt: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssistantConfig":
        """Создаёт конфигурацию из словаря."""
        config = cls()

        # Model config
        if "MODEL_CONFIG" in data:
            mc = data["MODEL_CONFIG"]
            gen = mc.get("generation", {})
            config.model = ModelConfig(
                provider=mc.get("provider", config.model.provider),
                model_name=mc.get("model_name", config.model.model_name),
                primary_model=mc.get("primary_model", config.model.primary_model),
                alternative_models=mc.get("alternative_models", []),
                temperature=gen.get("temperature", config.model.temperature),
                max_new_tokens=gen.get("max_new_tokens", config.model.max_new_tokens),
                top_p=gen.get("top_p", config.model.top_p),
                repetition_penalty=gen.get("repetition_penalty", config.model.repetition_penalty),
                do_sample=gen.get("do_sample", config.model.do_sample),
            )

        # Vector DB config
        if "VECTOR_DB" in data:
            vdb = data["VECTOR_DB"]
            emb = vdb.get("embeddings", {})
            config.vector_db = VectorDBConfig(
                type=vdb.get("type", config.vector_db.type),
                persist_directory=vdb.get("persist_directory", config.vector_db.persist_directory),
                collection_name=vdb.get("collection_name", config.vector_db.collection_name),
                embeddings=EmbeddingsConfig(
                    model=emb.get("model", config.vector_db.embeddings.model),
                    device=emb.get("device", config.vector_db.embeddings.device),
                ),
            )

        # RAG config
        if "RAG_CONFIG" in data:
            rag = data["RAG_CONFIG"]
            chunk = rag.get("chunking", {})
            retr = rag.get("retrieval", {})
            config.rag = RAGConfig(
                chunking=ChunkingConfig(
                    chunk_size=chunk.get("chunk_size", config.rag.chunking.chunk_size),
                    chunk_overlap=chunk.get("chunk_overlap", config.rag.chunking.chunk_overlap),
                    separators=chunk.get("separators", config.rag.chunking.separators),
                ),
                retrieval=RetrievalConfig(
                    search_type=retr.get("search_type", config.rag.retrieval.search_type),
                    k=retr.get("k", config.rag.retrieval.k),
                    fetch_k=retr.get("fetch_k", config.rag.retrieval.fetch_k),
                    lambda_mult=retr.get("lambda_mult", config.rag.retrieval.lambda_mult),
                    top_k_final=retr.get("top_k_final", config.rag.retrieval.top_k_final),
                ),
            )

        # Safety config
        if "SAFETY" in data:
            sf = data["SAFETY"].get("input_filter", {})
            config.safety = SafetyConfig(
                max_length=sf.get("max_length", config.safety.max_length),
                enabled=sf.get("enabled", config.safety.enabled),
            )

        # Tracing config
        if "TRACING" in data:
            tr = data["TRACING"]
            config.tracing = TracingConfig(
                enabled=tr.get("enabled", config.tracing.enabled),
                log_level=tr.get("log_level", config.tracing.log_level),
                log_file=tr.get("log_file", config.tracing.log_file),
            )

        # System prompt
        config.system_prompt = data.get("SYSTEM_PROMPT", "")

        return config


def get_default_config_path() -> Path:
    """Возвращает путь к default_config.yaml внутри пакета."""
    return Path(__file__).parent.parent / "default_config.yaml"


def load_config(config_path: Optional[str] = None) -> AssistantConfig:
    """
    Загружает конфигурацию из YAML файла.

    Args:
        config_path: Путь к файлу конфигурации.
                    Если не указан, использует default_config.yaml из пакета

    Returns:
        AssistantConfig с загруженными настройками
    """
    if config_path is None:
        # Используем default_config.yaml из пакета
        default_path = get_default_config_path()
        if default_path.exists():
            config_path = str(default_path)

    if config_path is None or not Path(config_path).exists():
        # Возвращаем конфигурацию по умолчанию
        return AssistantConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AssistantConfig.from_dict(data)
