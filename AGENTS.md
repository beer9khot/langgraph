# AGENTS Instructions

This repository is a monorepo for LangGraph—a low-level orchestration framework for building stateful, multi-actor agents. Each library lives in a subdirectory under `libs/`.

When you modify code in any library, run the following commands in that library's directory before creating a pull request:

- `make format` – run code formatters (ruff)
- `make lint` – run the linter (ruff + type checking)
- `make test` – execute the test suite (pytest)

To run a particular test file or pass additional pytest options, specify the `TEST` variable:

```
TEST=path/to/test.py make test
```

## Libraries Overview

The repository contains several Python and JavaScript/TypeScript libraries:

- **checkpoint** – base interfaces and protocols for LangGraph checkpointers
- **checkpoint-postgres** – Postgres implementation of checkpoint saver
- **checkpoint-sqlite** – SQLite implementation of checkpoint saver
- **checkpoint-conformance** – conformance tests for checkpoint implementations
- **cli** – official command-line interface for LangGraph
- **langgraph** – core framework for building stateful agents with graph execution
- **prebuilt** – high-level APIs for creating and running agents
- **sdk-js** – JS/TS SDK for interacting with the LangGraph REST API
- **sdk-py** – Python SDK for the LangGraph Server API

### Dependency Map

The diagram below shows downstream libraries for each production dependency:

```
checkpoint
├── checkpoint-postgres
├── checkpoint-sqlite
├── prebuilt
└── langgraph

prebuilt
└── langgraph

sdk-py
├── langgraph
└── cli

sdk-js (standalone)
```

Changes to a library impact all of its dependents shown above.

## Development Conventions

### Code Style

- **Imports**: Use absolute imports from `langgraph.*`; avoid relative imports. Alphabetical ordering with `typing_extensions` for backcompat
- **Future annotations**: Include `from __future__ import annotations` in all modules
- **Type hints**: Required everywhere—use TypedDict for state schemas, Pydantic models for validation, RunnableConfig for configs
- **Docstrings**: Pydantic models and public functions must have docstrings with summary, detailed description, Args, Returns, Raises sections. Use `!!! note` / `!!! warning` blocks for important context
- **Code comments**: Single backticks for inline code references (NOT double backticks)

### Architecture Patterns

- **Protocol-based composition**: Use Generic types and Protocol definitions instead of inheritance (see `BaseCheckpointSaver[V]`)
- **TypedDict state schemas**: Define state as `Annotated[Type, reducer_fn]` for multi-write aggregation
- **Pregel execution model**: Nodes are callables (functions or Runnable instances); channels manage state flow
- **Dual async/sync**: Checkpoint savers have sync variants (`__init__.py`) and async variants (`aio.py`)
- **Channels**: LastValue, DeltaChannel, Topic encapsulate state logic; versions tracked per-channel

### Testing

- **Framework**: pytest + pytest-asyncio
- **Infrastructure**: Docker Compose for Postgres/Redis services (required for checkpoint tests)
- **Test files**: Colocate tests in `tests/` directories with conftest.py fixtures
- **Snapshots**: Use syrupy via `__snapshots__/` directories for regression testing
- **Running tests**: Use `pytest -xvs` locally; for isolated runs without Docker: `NO_DOCKER=true make test`
- **Mocking**: Use pytest-mock; async mocks via unittest.mock.AsyncMock

### Dependency Management

- **Tool**: uv (replaces pip); generates uv.lock
- **Python version**: >=3.10 across all libraries; support through 3.13
- **Key dependencies**: 
  - Pydantic >=2.7.4 (mandatory for validation)
  - langchain-core >=1.4.7,<2 (langgraph)
  - langchain-core >=1.8.0,<2 (prebuilt)
- **Internal modules**: Files in `_internal/` subdirectories are unstable and not part of public API
- **Per-library versioning**: checkpoint-sqlite, checkpoint-postgres have independent version ranges to support different backends

## Common Patterns & Pitfalls

### State Management

- Define state schemas as TypedDict with `Annotated[Type, reducer_fn]` syntax
- Reducers are applied automatically; no explicit merge methods needed
- Use channels to encapsulate state logic (LastValue for most cases, Topic for pub/sub)

### Graph Execution

- Nodes return state updates as dict or StateUpdate object
- Edges route based on node output; use conditional edges for branching
- Subgraphs enabled via nested graph compiles
- Use Send() and Command() for dynamic fan-out to multiple nodes

### Checkpointing

- `BaseCheckpointSaver[V]` protocol requires: `get_tuple()`, `put()`, `list()`, `get_delta_channel_history()`
- Thread-safety via `thread_id` in config; namespace isolation via `checkpoint_ns`
- Delta snapshots for efficient incremental storage (see `counters_since_delta_snapshot` metadata)
- Security: Set `LANGGRAPH_STRICT_MSGPACK=true` or pass `allowed_msgpack_modules` list to prevent code execution from untrusted DBs

### Critical Pitfalls to Avoid

1. **Async/sync mismatch**: `SqliteSaver` is sync-only; use `AsyncSqliteSaver` for async code
2. **Import paths**: Always use absolute imports; relative imports cause path resolution issues
3. **State schema validation**: Invalid TypedDict/Pydantic schemas cause silent failures
4. **Serialization security**: Untrusted checkpoints can execute arbitrary code if not properly validated
5. **Version bounds**: langgraph requires langchain-core <2; prebuilt requires langgraph-checkpoint >=2.1.0,<5.0.0

## Error Handling

- Use `ErrorCode` enum in [langgraph/errors.py](libs/langgraph/langgraph/errors.py)
- Error documentation available at docs.langchain.com/errors/{code}
- Nested error handlers via node-level retry/error_handler policies
- Use `ParentCommand` for parent-graph control signals

## Documentation & Resources

- Official docs: [docs.langchain.com/oss/python/langgraph](https://docs.langchain.com/oss/python/langgraph)
- JS/TS library: [LangGraph.js](https://github.com/langchain-ai/langgraphjs)
- Checkpoint implementations: See [libs/checkpoint/README.md](libs/checkpoint/README.md)
- CLI guide: See [libs/cli/README.md](libs/cli/README.md)
