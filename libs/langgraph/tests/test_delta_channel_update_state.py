"""Tests for `update_state` / `aupdate_state` against `DeltaChannel`.

Regression suite for deepagents#3774 and Postgres read-path compatibility:
fresh-thread `update_state` must persist DeltaChannel state correctly.

Fresh-thread updates snapshot updated DeltaChannels on the head checkpoint
(self-contained for Postgres readers). Delta writes are not persisted via
`put_writes`; non-delta channel writes on a fresh thread are attached to
the head after it is saved.

Coverage:

* fresh-thread regression: single `update_state` writes a message and reads back
* non-fresh thread: `update_state` after `invoke`, after another `update_state`,
  and `bulk_update_state` with multiple per-superstep updates
* update-by-id end-to-end via `update_state` (DeltaChannel reducer semantics)
* state-history shape on a fresh thread (single snapshotted head checkpoint)
* head checkpoint snapshots updated DeltaChannels for Postgres read paths
"""

from typing import Annotated, Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.graph.message import _messages_delta_reducer

pytestmark = pytest.mark.anyio


def _build_graph(checkpointer: InMemorySaver, *, two_nodes: bool = False) -> Any:
    """Compile a minimal DeltaChannel-backed `messages` graph.

    `two_nodes=True` adds a second writer node so `bulk_update_state` can route
    distinct updates to different `as_node` values within a single superstep.
    """
    channel = DeltaChannel(_messages_delta_reducer)
    State = TypedDict("State", {"messages": Annotated[list, channel]})  # type: ignore[call-overload]  # noqa: UP013

    def model(state: dict) -> dict:
        return {}

    def assistant(state: dict) -> dict:
        return {}

    builder = StateGraph(State)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    if two_nodes:
        builder.add_node("assistant", assistant)
        builder.add_edge("model", "assistant")
        builder.set_finish_point("assistant")
    else:
        builder.set_finish_point("model")
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Fresh-thread regression (deepagents#3774)
# ---------------------------------------------------------------------------


def test_update_state_fresh_thread_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "fresh-sync"}}
    message = HumanMessage(content="hello", id="m1")

    graph.update_state(config, {"messages": [message]}, as_node="model")

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["hello"]


async def test_aupdate_state_fresh_thread_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "fresh-async"}}
    message = HumanMessage(content="hello", id="m1")

    await graph.aupdate_state(config, {"messages": [message]}, as_node="model")

    state = await graph.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["hello"]


# ---------------------------------------------------------------------------
# Non-fresh thread: update_state after invoke
# ---------------------------------------------------------------------------


def test_update_state_after_invoke_delta_channel() -> None:
    """The non-fresh-thread path must keep working across snapshot changes."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "after-invoke-sync"}}

    graph.invoke({"messages": [HumanMessage(content="seed", id="m1")]}, config)
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="appended", id="m2")]},
        as_node="model",
    )

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["seed", "appended"]
    assert [m.id for m in state.values["messages"]] == ["m1", "m2"]


async def test_aupdate_state_after_invoke_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "after-invoke-async"}}

    await graph.ainvoke({"messages": [HumanMessage(content="seed", id="m1")]}, config)
    await graph.aupdate_state(
        config,
        {"messages": [HumanMessage(content="appended", id="m2")]},
        as_node="model",
    )

    state = await graph.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["seed", "appended"]


# ---------------------------------------------------------------------------
# Non-fresh thread: consecutive update_state calls
# ---------------------------------------------------------------------------


def test_consecutive_update_states_delta_channel() -> None:
    """Two consecutive fresh-thread-style updates: the first creates a
    snapshotted head; the second anchors on it. Both messages round-trip."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "consecutive-sync"}}

    graph.update_state(
        config,
        {"messages": [HumanMessage(content="first", id="m1")]},
        as_node="model",
    )
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="second", id="m2")]},
        as_node="model",
    )

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["first", "second"]
    assert [m.id for m in state.values["messages"]] == ["m1", "m2"]


async def test_aconsecutive_update_states_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "consecutive-async"}}

    await graph.aupdate_state(
        config,
        {"messages": [HumanMessage(content="first", id="m1")]},
        as_node="model",
    )
    await graph.aupdate_state(
        config,
        {"messages": [HumanMessage(content="second", id="m2")]},
        as_node="model",
    )

    state = await graph.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["first", "second"]


# ---------------------------------------------------------------------------
# Update-by-id semantics through the update_state path
# ---------------------------------------------------------------------------


def test_update_state_replaces_message_by_id_delta_channel() -> None:
    """`_messages_delta_reducer` dedups by `id` — re-issuing a write with the
    same id replaces the existing entry rather than appending. Verify this
    works through the `update_state` path (not just `invoke`)."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "update-by-id"}}

    graph.invoke({"messages": [HumanMessage(content="original", id="h1")]}, config)
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="updated", id="h1")]},
        as_node="model",
    )

    state = graph.get_state(config)
    msgs = state.values["messages"]
    assert len(msgs) == 1
    assert msgs[0].id == "h1"
    assert msgs[0].content == "updated"


# ---------------------------------------------------------------------------
# bulk_update_state with multiple updates per superstep
# ---------------------------------------------------------------------------


def test_bulk_update_state_multi_task_per_superstep_delta_channel() -> None:
    """`bulk_update_state` with N updates in one superstep must accumulate
    all N message writes in the snapshotted head state.

    Explicit `task_id`s are required to disambiguate writes belonging to
    different `StateUpdate`s targeting the same node — otherwise both share
    the deterministic interrupt-derived id and collide in the saver.
    """
    from langgraph.types import StateUpdate

    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "bulk-multi-task"}}

    graph.bulk_update_state(
        config,
        [
            [
                StateUpdate(
                    values={"messages": [HumanMessage(content="first", id="m1")]},
                    as_node="model",
                    task_id="task-1",
                ),
                StateUpdate(
                    values={"messages": [HumanMessage(content="second", id="m2")]},
                    as_node="model",
                    task_id="task-2",
                ),
            ]
        ],
    )

    state = graph.get_state(config)
    contents = [m.content for m in state.values["messages"]]
    ids = [m.id for m in state.values["messages"]]
    assert sorted(contents) == ["first", "second"], (
        f"both updates' writes must persist; got {contents}"
    )
    assert sorted(ids) == ["m1", "m2"]


# ---------------------------------------------------------------------------
# Public-API observation of fresh-thread checkpoint shape
# ---------------------------------------------------------------------------


def test_state_history_chain_after_fresh_update_state_delta_channel() -> None:
    """Fresh-thread `update_state` on snapshotted DeltaChannels yields one
    self-contained checkpoint (step=0, no parent, source='update')."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "history-chain"}}

    graph.update_state(
        config,
        {"messages": [HumanMessage(content="hello", id="m1")]},
        as_node="model",
    )

    history = list(graph.get_state_history(config))
    assert len(history) == 1

    update_snapshot = history[0]

    assert update_snapshot.metadata is not None
    assert update_snapshot.metadata["source"] == "update"
    assert update_snapshot.metadata["step"] == 0
    assert update_snapshot.parent_config is None
    assert [m.content for m in update_snapshot.values["messages"]] == ["hello"]


def test_fresh_update_state_head_snapshots_delta_channel() -> None:
    """Postgres checkpointers skip the ancestor walk when the head checkpoint
    has no `counters_since_delta_snapshot` entry. Force-snapshot updated
    DeltaChannels on the update checkpoint so the head is self-contained."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "head-snapshot"}}

    graph.update_state(
        config,
        {"messages": [HumanMessage(content="hello", id="m1")]},
        as_node="model",
    )

    head = saver.get_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)
    assert [m.content for m in head.checkpoint["channel_values"]["messages"].value] == [
        "hello"
    ]
