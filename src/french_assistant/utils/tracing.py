"""
Модуль трассировки для отслеживания работы pipeline.

Предоставляет:
- TraceEvent: датакласс для событий трассировки
- TracingManager: менеджер для записи и анализа событий
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Событие трассировки для отладки и анализа."""

    timestamp: datetime
    event_type: str
    component: str
    input_data: Any
    output_data: Any
    metadata: Dict = field(default_factory=dict)
    duration_ms: float = 0.0


class TracingManager:
    """
    Менеджер трассировки для отслеживания всего pipeline.

    Пример использования:
        tracer = TracingManager()
        tracer.log_event("query_received", "Assistant", query, "processing")
        # ... выполнение операций ...
        print(tracer.get_trace_report())
    """

    def __init__(self):
        self.events: List[TraceEvent] = []
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

    def log_event(
        self,
        event_type: str,
        component: str,
        input_data: Any,
        output_data: Any,
        metadata: Dict = None,
        duration_ms: float = 0.0
    ) -> None:
        """
        Записывает событие трассировки.

        Args:
            event_type: Тип события (например, "query_received", "retrieval_complete")
            component: Компонент, генерирующий событие
            input_data: Входные данные
            output_data: Выходные данные
            metadata: Дополнительные метаданные
            duration_ms: Длительность операции в миллисекундах
        """
        event = TraceEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            component=component,
            input_data=str(input_data)[:500],
            output_data=str(output_data)[:500],
            metadata=metadata or {},
            duration_ms=duration_ms
        )
        self.events.append(event)
        logger.debug(f"[TRACE][{component}] {event_type}: {str(input_data)[:100]}...")

    def get_trace_report(self) -> str:
        """Генерирует текстовый отчёт о трассировке."""
        report = [
            f"\n{'=' * 60}",
            f"TRACE REPORT - Session: {self.session_id}",
            f"{'=' * 60}"
        ]

        for event in self.events:
            report.append(
                f"\n[{event.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
                f"{event.component} -> {event.event_type}"
            )
            report.append(f"  Input: {event.input_data[:100]}...")
            report.append(f"  Output: {event.output_data[:100]}...")
            if event.duration_ms > 0:
                report.append(f"  Duration: {event.duration_ms:.2f}ms")

        return "\n".join(report)

    def get_events_by_component(self, component: str) -> List[TraceEvent]:
        """Возвращает события для указанного компонента."""
        return [e for e in self.events if e.component == component]

    def get_total_duration(self) -> float:
        """Возвращает общую длительность всех операций."""
        return sum(e.duration_ms for e in self.events)

    def clear(self) -> None:
        """Очищает историю событий."""
        self.events.clear()
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
