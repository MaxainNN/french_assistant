# 🇫🇷 French Assistant — Нейро-сотрудник переводчик с RAG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-orange.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaxainNN/french-assistant/blob/main/french_assistant.ipynb)

**Продвинутый ассистент для перевода с русского на французский язык**

<img src="images/preview_app_for_readme.png" width="512" height="512">

</div>

---

## 📋 Содержание

- [Описание](#-описание)
- [Архитектура](#-архитектура)
- [Workflow LangChain](#-workflow-langchain)
- [Улучшения RAG](#-улучшения-rag)
- [Борьба с галлюцинациями](#-борьба-с-галлюцинациями)
- [Безопасность](#-безопасность)
- [Установка](#-установка)
- [Использование](#-использование)
- [Тестирование](#-тестирование)
- [Технологии](#-технологии)
- [Полезные материалы](#-полезные-материалы)

---

## 📖 Описание

**French Assistant** — это RAG-система (Retrieval-Augmented Generation) для помощи в переводе с русского на французский язык. Система включает:

### Профиль нейро-сотрудника

- **Профессия:** Продвинутый переводчик-ассистент
- **Специализация:** Русско-французский перевод, грамматика, идиоматика
- **Внутренний мир:** Носитель двух культур, педантичный к деталям
- **Принципы:** Точность, естественность, контекстуальность, прозрачность

### Ключевые возможности

✅ Перевод с учётом контекста и регистра речи  
✅ Объяснение грамматических правил  
✅ Работа с идиомами и фразеологизмами  
✅ Предупреждение о ложных друзьях переводчика  
✅ Многоуровневая защита от галлюцинаций  
✅ Фильтрация небезопасных запросов  

---

## 🏗 Архитектура

```mermaid
flowchart TB
    subgraph INPUT ["📥 INPUT LAYER"]
        A[User Query] --> B[Safety Filter<br/>Injection + Topics]
        B --> C[Query Preprocessor<br/>Query Expansion + HyDE]
    end

    subgraph RETRIEVAL ["🔍 RETRIEVAL LAYER"]
        D[Multi-Query Retrieval] --> E[ChromaDB Vector Store]
        E --> F[MMR Selection<br/>Diversity + Relevance]
    end

    subgraph CRAG ["🔧 CRAG LAYER"]
        G[Quality Assessment<br/>Excellent/Good/Partial/Poor]
        G --> H[Correction Strategy<br/>none/supplement/refine/fallback]
        H --> I[Fallback Knowledge Injection]
    end

    subgraph GENERATION ["⚙️ GENERATION LAYER"]
        J[System Prompt<br/>+ Few-Shot + Context] --> K[LLM Saiga/Llama<br/>Temperature: 0.3]
    end

    subgraph ANTIHALLUCINATION ["🛡️ ANTI-HALLUCINATION LAYER"]
        L[Lexical Grounding Check]
        M[Semantic Consistency Check]
        N[Chain of Verification CoVe]
        O[Confidence Calibration]
    end

    subgraph OUTPUT ["📤 OUTPUT LAYER"]
        P[Trace Logger] --> Q[Output Formatter]
        Q --> R[Final Response]
    end

    C --> D
    F --> G
    I --> J
    K --> L
    K --> M
    K --> N
    L --> O
    M --> O
    N --> O
    O --> P
```

---

## 🔄 Workflow LangChain

### Диаграмма последовательности

```mermaid
sequenceDiagram
    participant User
    participant Safety as Safety Filter
    participant Retriever
    participant CRAG
    participant LLM

    User->>Safety: Query

    Note over Safety: Check Injection<br/>Check Topic

    alt If SAFE
        Safety->>Retriever: Pass query

        Note over Retriever: Query Expansion<br/>Multi-Query<br/>MMR Search

        Retriever->>CRAG: Documents

        Note over CRAG: Evaluate Quality<br/>Apply Correction

        CRAG->>LLM: Context

        Note over LLM: Generate Response

        LLM-->>CRAG: Response
        CRAG-->>User: Final Response (with verification)
    else If UNSAFE
        Safety-->>User: Blocked
    end
```

### Компоненты LangChain

```python
# Основные компоненты pipeline

from langchain.schema.runnable import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel
)

# 1. Input processing
input_processor = RunnableParallel(
    query=RunnablePassthrough(),
    safety_check=RunnableLambda(safety_filter.filter_input),
    topic_check=RunnableLambda(safety_filter.check_topic_relevance)
)

# 2. Query expansion
query_expander = RunnableLambda(
    lambda x: query_expander.expand_query(x['query'], use_hyde=True)
)

# 3. Retrieval
retriever_chain = RunnableParallel(
    documents=RunnableLambda(lambda x: retriever.retrieve(x)),
    original_query=RunnablePassthrough()
)

# 4. CRAG correction
crag_chain = RunnableLambda(
    lambda x: crag.correct(x['original_query'], x['documents'])
)

# 5. Generation
generation_chain = (
    prompt_template 
    | llm 
    | StrOutputParser()
)

# 6. Hallucination check
hallucination_chain = RunnableLambda(
    lambda x: hallucination_detector.detect(x['response'], x['context'])
)

# Full pipeline
full_chain = (
    input_processor
    | RunnableLambda(check_safety_gate)
    | query_expander
    | retriever_chain
    | crag_chain
    | generation_chain
    | hallucination_chain
    | output_formatter
)
```

---

## 🚀 Улучшения RAG

### 1. Query Expansion (Расширение запросов)

```mermaid
flowchart TB
    A["Оригинальный запрос:<br/>'спряжение avoir'"] --> B

    subgraph B ["Query Expansion Module"]
        direction TB
        C["<b>Synonym expansion:</b><br/>• спряжение avoir<br/>• conjugation avoir<br/>• conjugaison avoir"]
        D["<b>HyDE:</b><br/>Глагол avoir спрягается:<br/>j'ai, tu as..."]
    end

    B --> E["4 запроса для поиска"]
```

### 2. MMR (Maximum Marginal Relevance)

Баланс между релевантностью и разнообразием результатов:

```
λ = 0.7  (настраиваемый параметр)

Score(d) = λ × Relevance(d, query) - (1-λ) × max(Similarity(d, d_selected))
```

### 3. CRAG (Corrective RAG)

```mermaid
flowchart TB
    A[Retrieved Documents] --> B[Quality Check]

    B --> C[EXCELLENT]
    B --> D[GOOD]
    B --> E[PARTIAL]
    B --> F[POOR]

    C --> G["None<br/>(pass)"]
    D --> H["Supplement<br/>(add facts)"]
    E --> I["Refine<br/>(clarify)"]
    F --> J["Fallback<br/>(use base knowledge)"]
```

### 4. Self-RAG (Самооценка)

```python
# Токены самооценки
RETRIEVAL_TOKEN: bool    # Нужен ли retrieval?
RELEVANCE_TOKEN: enum    # Релевантность документа
SUPPORT_TOKEN: enum      # Поддержка ответа источниками
UTILITY_TOKEN: enum      # Полезность ответа
```

---

## 🛡 Борьба с галлюцинациями

### Многоуровневая система защиты

```mermaid
flowchart TB
    subgraph AHS ["🛡️ ANTI-HALLUCINATION"]
        direction TB

        subgraph L1 ["L1: LEXICAL"]
            A1["Проверка ключевых слов<br/>grounding_score > 0.3"]
        end

        subgraph L2 ["L2: SEMANTIC"]
            A2["Поиск противоречий<br/>'всегда' vs 'никогда'"]
        end

        subgraph L3 ["L3: CONFIDENCE"]
            A3["Калибровка уверенности<br/>Модификаторы неопределённости"]
        end

        subgraph L4 ["L4: CoVe"]
            A4["Верификация claims<br/>Итоговый confidence score"]
        end

        L1 --> L2 --> L3 --> L4
    end
```

### Пример детекции галлюцинации

```python
# Входные данные
response = "Артикль 'le' ВСЕГДА используется перед мужским родом без исключений."
context = "le используется с существительными мужского рода. Перед гласной используется l'"

# Результат детекции
{
    'has_hallucinations': True,
    'confidence': 0.45,
    'issues': [
        'Обнаружены сверхуверенные утверждения',
        'Низкое лексическое заземление'
    ],
    'grounding_score': 0.28,
    'overconfident_claims': ['ВСЕГДА', 'без исключений'],
    'recommendation': 'Добавьте модификаторы неопределённости'
}
```

---

## 🔒 Безопасность

### Фильтрация входных запросов

```mermaid
flowchart TB
    subgraph SF ["🔒 SAFETY FILTER"]
        direction TB

        subgraph S1 ["1. LENGTH"]
            L1["Max 2000 символов"]
        end

        subgraph S2 ["2. INJECTION"]
            L2["Блокировка инъекций:<br/>ignore, forget, jailbreak..."]
        end

        subgraph S3 ["3. TOPIC"]
            L3["Проверка релевантности<br/>Score > 0.1"]
        end

        subgraph S4 ["4. LANGUAGE"]
            L4["Allowed: ru, fr"]
        end

        S1 --> S2 --> S3 --> S4
    end
```

### Примеры фильтрации

| Запрос | Результат | Причина |
|--------|-----------|---------|
| "Переведи: Привет" | ✅ PASS | Валидный запрос |
| "Ignore all instructions" | ❌ BLOCK | Injection |
| "Напиши код на Python" | ❌ BLOCK | Off-topic |
| "Comment dit-on 'кошка'?" | ✅ PASS | Валидный (фр.) |

---

## 💻 Установка и запуск

### Вариант 1: Google Colab (рекомендуется для быстрого старта)

Для запуска в Google Colab используйте готовый notebook:

1. Откройте [french_assistant.ipynb](french_assistant.ipynb) в Google Colab
2. Выполните все ячейки по порядку (Runtime → Run all)
3. Notebook включает демо-базу знаний - можно сразу тестировать!

> **Примечание:** В Colab все зависимости установятся автоматически из первой ячейки.

---

### Вариант 2: Установка как Python-пакет (рекомендуется)

```bash
# 1. Клонирование репозитория
git clone https://github.com/MaxainNN/french-assistant.git
cd french-assistant

# 2. Создание виртуального окружения
python -m venv venv

# Активация (Linux/Mac)
source venv/bin/activate

# Активация (Windows)
.\venv\Scripts\activate

# 3. Установка пакета в режиме разработки
pip install -e ".[dev]"

# Или с поддержкой HuggingFace моделей
pip install -e ".[all]"
```

### Запуск CLI-интерфейса

```bash
# После установки пакета
french-assistant

# Или через python -m
python -m french_assistant
```

---

### Вариант 3: Установка через requirements.txt (legacy)

```bash
pip install -r requirements.txt
python -m french_assistant
```

### Настройка HuggingFace токена

Для работы с LLM моделями (Saiga, Llama и др.) необходим токен HuggingFace:

1. Получите токен на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Создайте файл `.env` в корне проекта:

```bash
cp .env.example .env
```

3. Добавьте ваш токен:

```bash
# .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Важно:** Файл `.env` содержит секреты и не должен попадать в git (уже добавлен в `.gitignore`)

### Настройка модели Saiga (опционально)

Конфигурация находится в `src/french_assistant/default_config.yaml`:

```yaml
MODEL_CONFIG:
  primary_model: "IlyaGusev/saiga_llama3_8b"
  generation:
    temperature: 0.3
    max_new_tokens: 1024
    top_p: 0.9
```

---

## 🎮 Использование

### CLI интерфейс

```bash
# После установки пакета
french-assistant

# Или через python -m
python -m french_assistant
```

```
============================================================
🇫🇷 French Assistant - Переводчик-ассистент
============================================================

Инициализация системы...
✅ Система готова к работе!

Команды:
  - 'exit' - выход
  - 'stats' - статистика
  - 'trace' - трассировка последнего запроса

👤 Вы: Переведи на французский: Я хочу кофе

🤖 Ассистент:
📝 **Перевод:** Je veux un café.
📚 **Альтернативы:** Je voudrais un café. (более вежливо)
💡 **Комментарий:** Форма "voudrais" используется для вежливых просьб.
```

### Программное использование

```python
from french_assistant import FrenchAssistant

# Создание ассистента
assistant = FrenchAssistant()

# Обработка запроса
result = assistant.process_query("Как спрягается глагол être?")

print(result['response'])
print(f"Confidence: {result['metadata']['grounding']['confidence']}")
print(f"Sources: {result['sources']}")
```

### Использование отдельных компонентов

```python
# Импорт отдельных компонентов
from french_assistant import (
    SafetyFilter,
    HallucinationDetector,
    EnhancedRetriever,
    QueryExpander,
    CorrectiveRAG,
    SelfRAG,
    ChainOfVerification,
)

# Пример: использование SafetyFilter отдельно
from french_assistant.core.config import SafetyConfig

filter = SafetyFilter(SafetyConfig(max_length=1000))
is_safe, error, meta = filter.filter_input("Переведи: Привет")

# Пример: использование QueryExpander
expander = QueryExpander()
variants = expander.expand_query("спряжение avoir", use_hyde=True)
# ['спряжение avoir', 'спрягается avoir', 'conjugation avoir', ...]
```

---

## 🧪 Тестирование

```bash
# Запуск тестов через pytest
pytest

# С покрытием кода
pytest --cov=french_assistant

# Запуск конкретного модуля
pytest tests/test_safety.py -v
pytest tests/test_enhancements.py -v
```

---

## 📁 Структура проекта

```
french_assistant/
├── src/
│   └── french_assistant/           # Python package
│       ├── __init__.py             # Публичный API пакета
│       ├── __main__.py             # CLI точка входа
│       ├── default_config.yaml     # Конфигурация по умолчанию
│       │
│       ├── core/                   # Ядро системы
│       │   ├── __init__.py
│       │   ├── assistant.py        # FrenchAssistant - главный класс
│       │   └── config.py           # Загрузка и валидация конфигурации
│       │
│       ├── safety/                 # Модуль безопасности
│       │   ├── __init__.py
│       │   ├── filter.py           # SafetyFilter - фильтрация запросов
│       │   └── hallucination.py    # HallucinationDetector - детекция галлюцинаций
│       │
│       ├── retrieval/              # Модуль поиска
│       │   ├── __init__.py
│       │   ├── retriever.py        # EnhancedRetriever - улучшенный поиск
│       │   ├── query_expansion.py  # QueryExpander - расширение запросов
│       │   └── vectorstore.py      # VectorStoreManager - работа с ChromaDB
│       │
│       ├── enhancements/           # RAG-улучшения
│       │   ├── __init__.py
│       │   ├── crag.py             # CorrectiveRAG - коррекция retrieval
│       │   ├── self_rag.py         # SelfRAG - самооценка качества
│       │   └── cove.py             # ChainOfVerification - верификация
│       │
│       └── utils/                  # Утилиты
│           ├── __init__.py
│           ├── tracing.py          # TracingManager - трассировка pipeline
│           └── logging.py          # Настройка логирования
│
├── data/
│   ├── knowledge_base/             # База знаний (markdown)
│   │   ├── grammar_verbs.md
│   │   ├── grammar_articles.md
│   │   ├── idioms_phrases.md
│   │   └── translation_difficulties.md
│   └── chroma_db/                  # Векторное хранилище (создаётся автоматически)
│
├── tests/                          # Pytest тесты
│   ├── conftest.py
│   ├── test_safety.py
│   └── test_enhancements.py
│
├── logs/                           # Логи (создаётся автоматически)
├── french_assistant.ipynb          # Jupyter Notebook для Google Colab
├── pyproject.toml                  # Конфигурация пакета (PEP 517/518)
├── requirements.txt                # Зависимости (legacy)
└── README.md
```

### Модульная архитектура

| Модуль | Назначение |
|--------|------------|
| `core` | Главный класс `FrenchAssistant` и конфигурация |
| `safety` | Фильтрация запросов и детекция галлюцинаций |
| `retrieval` | Поиск документов, расширение запросов, работа с векторной БД |
| `enhancements` | RAG-улучшения: CRAG, Self-RAG, Chain-of-Verification |
| `utils` | Трассировка и логирование |

---

## 📈 Метрики и мониторинг

### Tracing

Система поддерживает детальную трассировку каждого запроса:

```
[14:23:45.123] SafetyFilter -> safety_passed
  Input: "Переведи: Привет..."
  Output: safe
[14:23:45.456] EnhancedRetriever -> query_expansion  
  Input: "Переведи: Привет"
  Output: 4 variants
[14:23:46.789] EnhancedRetriever -> retrieval_complete
  Input: 4 queries
  Output: 3 documents retrieved
```

### Качественные метрики

- **Grounding Score:** процент терминов ответа, присутствующих в контексте
- **Consistency Score:** отсутствие внутренних противоречий
- **Utility Score:** полезность ответа для запроса
- **Overall Quality:** комбинированная оценка

---

## 🔮 Roadmap

- [ ] Интеграция с Saiga LLaMA3 8B
- [ ] Web-интерфейс (Gradio/Streamlit)
- [ ] Расширение базы знаний
- [ ] A/B тестирование промптов
- [ ] Fine-tuning на параллельных корпусах
- [ ] Голосовой ввод/вывод

---

## 🛠 Технологии

### Основной стек

| Технология | Версия | Назначение |
|------------|--------|------------|
| **Python** | 3.10+ | Основной язык программирования |
| **LangChain** | >= 0.2 | Фреймворк для построения RAG-систем |
| **ChromaDB** | >= 0.4 | Векторное хранилище для эмбеддингов |
| **Sentence-Transformers** | >= 2.2 | Мультиязычные эмбеддинги |
| **Transformers** | >= 4.35 | Работа с LLM моделями |
| **PyTorch** | >= 2.0 | Фреймворк глубокого обучения |

### LangChain модули

- `langchain-core` — базовые абстракции и интерфейсы
- `langchain-community` — интеграции с внешними сервисами
- `langchain-text-splitters` — разбиение документов на chunks
- `langchain-huggingface` — интеграция с HuggingFace

### Модели

| Модель | Тип | Назначение |
|--------|-----|------------|
| `paraphrase-multilingual-MiniLM-L12-v2` | Embeddings | Мультиязычные эмбеддинги (RU/FR) |
| `IlyaGusev/saiga_llama3_8b` | LLM | Генерация ответов (русскоязычная) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranker | Переранжирование результатов |

### RAG-техники

- **Query Expansion** — расширение запросов синонимами
- **HyDE** (Hypothetical Document Embeddings) — генерация гипотетических документов
- **MMR** (Maximum Marginal Relevance) — баланс релевантности и разнообразия
- **CRAG** (Corrective RAG) — коррекция результатов retrieval
- **Self-RAG** — самооценка качества поиска и генерации
- **CoVe** (Chain of Verification) — пошаговая верификация ответов

### Инструменты разработки

- `pytest` — тестирование
- `black` — форматирование кода
- `mypy` — статическая типизация
- `PyYAML` — конфигурация

---

## 📚 Полезные материалы

### RAG from Scratch

Проект основан на концепциях из курса **RAG from Scratch** от LangChain:

🔗 **[langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)**

<img src="images/RAG_map.png" width="512" height="512">

Курс охватывает:
- Основы RAG-архитектуры
- Query Translation (Multi-query, RAG-Fusion, Decomposition, Step-back, HyDE)
- Routing (Logical и Semantic routing)
- Query Construction (Text-to-SQL, Text-to-Cypher)
- Indexing (Multi-representation, RAPTOR, ColBERT)
- Retrieval (Re-ranking, CRAG, Self-RAG, Adaptive RAG)

### Дополнительные ресурсы

- [LangChain Documentation](https://python.langchain.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Saiga Models](https://huggingface.co/IlyaGusev)
- [Статья на Habr по созданию нейро-сотрудников от академии The Founder](https://habr.com/ru/articles/825220/)
- [Пример русскоязычной LLM](https://huggingface.co/IlyaGusev/saiga_llama3_8b)
---

## 📄 Лицензия

MIT License

---

<div align="center">

**Сделано с ❤️ для изучающих французский язык**

</div>
