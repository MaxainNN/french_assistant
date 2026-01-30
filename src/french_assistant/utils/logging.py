"""
Модуль настройки логирования.

Предоставляет единую точку конфигурации логирования для всего пакета.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Настраивает логирование для пакета french_assistant.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу логов (опционально)
        log_format: Формат сообщений (опционально)

    Returns:
        Настроенный корневой логгер пакета

    Пример:
        from french_assistant.utils.logging import setup_logging
        logger = setup_logging(log_level="DEBUG", log_file="logs/app.log")
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Получаем корневой логгер пакета
    logger = logging.getLogger("french_assistant")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Очищаем существующие хэндлеры
    logger.handlers.clear()

    # Консольный хэндлер
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # Файловый хэндлер (если указан путь)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Получает логгер для указанного модуля.

    Args:
        name: Имя модуля (обычно __name__)

    Returns:
        Логгер для модуля

    Пример:
        from french_assistant.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Сообщение")
    """
    return logging.getLogger(f"french_assistant.{name}")
