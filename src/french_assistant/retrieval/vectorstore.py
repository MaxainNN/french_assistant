"""
Модуль работы с векторным хранилищем.

Предоставляет:
- VectorStoreManager: управление ChromaDB
- Загрузка и индексация документов
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma

# HuggingFace Embeddings с fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from ..core.config import VectorDBConfig, ChunkingConfig

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Менеджер векторного хранилища.

    Управляет:
    - Инициализацией эмбеддингов
    - Созданием/загрузкой ChromaDB
    - Загрузкой и индексацией документов

    Пример использования:
        manager = VectorStoreManager(config)
        vectorstore = manager.get_vectorstore()
        retriever = vectorstore.as_retriever()
    """

    def __init__(
        self,
        config: VectorDBConfig = None,
        chunking_config: ChunkingConfig = None
    ):
        """
        Инициализирует менеджер векторного хранилища.

        Args:
            config: Конфигурация векторной БД
            chunking_config: Конфигурация разбиения документов
        """
        self.config = config or VectorDBConfig()
        self.chunking_config = chunking_config or ChunkingConfig()

        self._embeddings = None
        self._vectorstore = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Возвращает инициализированные эмбеддинги (lazy loading)."""
        if self._embeddings is None:
            self._embeddings = self._init_embeddings()
        return self._embeddings

    @property
    def vectorstore(self) -> Chroma:
        """Возвращает векторное хранилище (lazy loading)."""
        if self._vectorstore is None:
            self._vectorstore = self._init_vectorstore()
        return self._vectorstore

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Инициализирует мультиязычные эмбеддинги."""
        model_name = self.config.embeddings.model
        device = self.config.embeddings.device

        logger.info(f"Loading embeddings model: {model_name}")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

    def _init_vectorstore(self) -> Chroma:
        """Инициализирует или загружает векторное хранилище."""
        persist_dir = self.config.persist_directory
        collection_name = self.config.collection_name

        # Проверяем, существует ли уже база
        if os.path.exists(persist_dir):
            logger.info(f"Loading existing vectorstore from {persist_dir}")
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )

        # Создаём новую базу
        logger.info("Creating new vectorstore...")
        documents = self.load_knowledge_base()

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name
        )

        logger.info(f"Vectorstore created with {len(documents)} documents")
        return vectorstore

    def load_knowledge_base(
        self,
        kb_path: str = "data/knowledge_base"
    ) -> List[Document]:
        """
        Загружает и обрабатывает базу знаний.

        Args:
            kb_path: Путь к директории с документами

        Returns:
            Список обработанных документов
        """
        # Загрузка markdown файлов
        loader = DirectoryLoader(
            kb_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        raw_docs = loader.load()
        logger.info(f"Loaded {len(raw_docs)} raw documents")

        # Разбиение на chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_config.chunk_size,
            chunk_overlap=self.chunking_config.chunk_overlap,
            separators=self.chunking_config.separators
        )

        documents = splitter.split_documents(raw_docs)

        # Добавляем метаданные
        for doc in documents:
            self._enrich_metadata(doc)

        logger.info(f"Split into {len(documents)} chunks with metadata")
        return documents

    def _enrich_metadata(self, doc: Document) -> None:
        """Обогащает документ метаданными."""
        source = doc.metadata.get("source", "")
        content = doc.page_content.lower()

        # Определяем тему по источнику
        if "grammar_verbs" in source:
            doc.metadata["topic"] = "grammar"
            doc.metadata["subtopic"] = "verbs"
        elif "grammar_articles" in source:
            doc.metadata["topic"] = "grammar"
            doc.metadata["subtopic"] = "articles"
        elif "idioms" in source:
            doc.metadata["topic"] = "idioms"
        elif "translation" in source:
            doc.metadata["topic"] = "translation"

        # Определяем сложность
        if "subjonctif" in content or "conditionnel" in content:
            doc.metadata["difficulty"] = "advanced"
        elif "passé composé" in content or "imparfait" in content:
            doc.metadata["difficulty"] = "intermediate"
        else:
            doc.metadata["difficulty"] = "beginner"

    def get_vectorstore(self) -> Chroma:
        """Возвращает векторное хранилище."""
        return self.vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """
        Добавляет документы в хранилище.

        Args:
            documents: Список документов для добавления
        """
        self.vectorstore.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vectorstore")

    def get_collection_count(self) -> int:
        """Возвращает количество документов в коллекции."""
        return self.vectorstore._collection.count()
