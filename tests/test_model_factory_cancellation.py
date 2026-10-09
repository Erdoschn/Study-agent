from __future__ import annotations

import threading
from types import SimpleNamespace

from core.model_factory import ModelClientFactory


def test_factory_binds_cancellation_event_only_for_active_run():
    factory = ModelClientFactory({})
    client = SimpleNamespace(cancellation_event=None)
    factory._browser_clients["deepseek-web"] = client
    event = threading.Event()

    factory.begin_run(event)
    assert client.cancellation_event is event
    assert factory._run_context.cancellation_event is event

    event.set()
    assert client.cancellation_event.is_set()

    factory.end_run(event)
    assert client.cancellation_event is None
    assert not hasattr(factory._run_context, "cancellation_event")


def test_factory_end_run_does_not_clear_a_newer_event():
    factory = ModelClientFactory({})
    first = threading.Event()
    second = threading.Event()
    client = SimpleNamespace(cancellation_event=None)
    factory._browser_clients["deepseek-web"] = client

    factory.begin_run(first)
    factory.begin_run(second)
    factory.end_run(first)

    assert factory._run_context.cancellation_event is second
    assert client.cancellation_event is second
