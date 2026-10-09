import threading
from threading import Event
from typing import Any

from .__debug__ import debug
from .model_registry import ModelInfo
from .reasoner import OpenAICompatibleClient


class ModelClientFactory:

    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config
        self._browser_clients: dict[str, Any] = {}
        self._run_context = threading.local()

    def begin_run(self, cancellation_event: Event) -> None:
        """Bind one request's cancellation event to browser clients on this thread."""
        self._run_context.cancellation_event = cancellation_event
        for client in tuple(self._browser_clients.values()):
            bind = getattr(client, "bind_cancellation_event", None)
            if callable(bind):
                bind(cancellation_event)
            else:
                client.cancellation_event = cancellation_event

    def end_run(self, cancellation_event: Event) -> None:
        """Detach a completed request's event so it cannot poison the next run."""
        if getattr(self._run_context, "cancellation_event", None) is cancellation_event:
            del self._run_context.cancellation_event
        for client in tuple(self._browser_clients.values()):
            bind = getattr(client, "bind_cancellation_event", None)
            if callable(bind):
                bind(None)
            elif getattr(client, "cancellation_event", None) is cancellation_event:
                client.cancellation_event = None

    def create(
        self,
        model: ModelInfo,
    ):
        with debug.scope(
            "ModelClientFactory",
            f"CREATE → {model.name}",
        ):
            providers = self.config.get(
                "providers",
                {},
            )

            provider = providers.get(
                model.provider
            )

            if not provider:
                raise RuntimeError(
                    f"找不到 Provider："
                    f"{model.provider}"
                )

            provider_type = provider.get(
                "type",
                "openai_compatible",
            )

            debug.log(
                "ModelClientFactory",
                f"provider={model.provider}, "
                f"type={provider_type}",
            )

            if provider_type == "browser":
                return self._create_browser_client(model, provider)

            if provider_type != (
                "openai_compatible"
            ):
                raise RuntimeError(
                    f"暂不支持 Provider 类型："
                    f"{provider_type}"
                )

            return OpenAICompatibleClient(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=model.model,
                timeout=int(provider.get("timeout", 120)),
                headers=provider.get("headers", {}),
            )

    def close(self) -> None:
        """Close cached browser-backed model clients."""
        for client in tuple(self._browser_clients.values()):
            try:
                client.close()
            except Exception as exc:
                debug.log(
                    "ModelClientFactory",
                    f"BROWSER CLOSE SKIP → {type(exc).__name__}: {exc}",
                )
        self._browser_clients.clear()

    def _create_browser_client(
        self,
        model: ModelInfo,
        provider: dict[str, Any],
    ):
        cancellation_event = getattr(
            self._run_context, "cancellation_event", None
        )
        cached = self._browser_clients.get(model.name)
        if cached is not None:
            cached.cancellation_event = cancellation_event
            return cached

        # Playwright's synchronous API is thread-bound. Browser-backed
        # models used by the threaded HTTP servers must live on a dedicated
        # worker thread instead of being created during prewarm and later
        # invoked from unrelated request-handler threads.
        from coder.browser_session import CoderBrowserSession

        response_selectors = provider.get("response_selectors")
        if isinstance(response_selectors, list):
            response_selectors = tuple(
                str(selector)
                for selector in response_selectors
                if str(selector).strip()
            )
        else:
            response_selectors = None

        client = CoderBrowserSession(
            model=model.model,
            url=provider.get("url"),
            user_data_dir=provider.get("user_data_dir"),
            browser_channel=provider.get("browser_channel"),
            timeout=int(provider.get("timeout", 180)),
            response_selectors=response_selectors,
            loading_selectors=provider.get("loading_selectors"),
            session_pause_seconds=float(provider.get("session_pause_seconds", 1.5)),
            cleanup_pause_seconds=float(provider.get("cleanup_pause_seconds", 3.0)),
            post_cleanup_pause_seconds=float(provider.get("post_cleanup_pause_seconds", 1.5)),
            cleanup_after_generate=bool(provider.get("cleanup_after_generate", True)),
            reuse_chat=bool(provider.get("reuse_chat", False)),
            min_send_interval_seconds=float(
                provider.get("min_send_interval_seconds", 5.0)
            ),
            debug_mode=False,
        )
        if cancellation_event is not None:
            client.bind_cancellation_event(cancellation_event)
        self._browser_clients[model.name] = client
        return client
