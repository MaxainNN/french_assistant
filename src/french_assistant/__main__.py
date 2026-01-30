"""
CLI интерфейс для French Assistant.

Запуск:
    python -m french_assistant
    # или после установки:
    french-assistant
"""

import yaml
from .core.assistant import FrenchAssistant
from .utils.logging import setup_logging


def main():
    """Точка входа для CLI."""
    print("\nИнициализация асистента...")
    setup_logging(log_level="INFO")

    try:
        assistant = FrenchAssistant()
        print("Асистент готова к работе!\n")
        print("Команды:")
        print("  - 'exit' - выход")
        print("  - 'stats' - статистика")
        print("  - 'trace' - трассировка последнего запроса")
        print("-" * 40)

        while True:
            try:
                query = input("\n👤 Вы: ").strip()
            except EOFError:
                break

            if query.lower() == "exit":
                print("\nДо свидания! Au revoir! 👋")
                break

            if query.lower() == "stats":
                stats = assistant.get_statistics()
                print(f"\n📊 Статистика:\n{yaml.dump(stats, allow_unicode=True)}")
                continue

            if query.lower() == "trace":
                print(assistant.tracer.get_trace_report())
                continue

            if not query:
                continue

            result = assistant.process_query(query)

            print(f"\n🤖 Ассистент:\n{result['response']}\n")

            if result["sources"]:
                print("📚 Источники:")
                for i, src in enumerate(result["sources"], 1):
                    topic = src["metadata"].get("topic", "unknown")
                    print(f"  {i}. [{topic}] {src['content'][:80]}...")

            print("-" * 40)

    except KeyboardInterrupt:
        print("\n\nПрервано пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
