import threading
from threading import Event
from pathlib import Path
from typing import Any

from .__debug__ import debug
from .model_registry import ModelInfo
from .reasoner import OpenAICompatibleClient
from .browser_automation_config import deep_merge_config
from .token_usage import TokenUsageRecorder


class ModelClientFactory:

    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config
        observability = config.get("observability", {})
        if not isinstance(observability, dict):
            observability = {}
        token_usage_config = observability.get("token_usage", {})
        self.token_usage_recorder = TokenUsageRecorder(
            token_usage_config if isinstance(token_usage_config, dict) else {},
            project_root=Path(__file__).resolve().parents[1],
        )
        self._browser_clients: dict[str, Any] = {}
        self._run_context = threading.local()

    def begin_run(self, cancellation_event: Event, attachments=None) -> None:
        """Bind cancellation and optional raw attachments to this Agent request."""
        self._run_context.cancellation_event = cancellation_event
        self._run_context.attachments = tuple(attachments or ())
        for client in tuple(self._browser_clients.values()):
            bind = getattr(client, "bind_cancellation_event", None)
            if callable(bind):
                bind(cancellation_event)
            else:
                client.cancellation_event = cancellation_event
            if self._run_context.attachments:
                begin_attachments = getattr(client, "begin_attachment_session", None)
                if not callable(begin_attachments):
                    raise RuntimeError(
                        "当前浏览器模型不支持直接上传附件；未将文件内容转换成提示词。"
                    )
                begin_attachments(self._run_context.attachments)

    def end_run(self, cancellation_event: Event) -> None:
        """Clean temporary browser attachments and detach request cancellation."""
        attachments = getattr(self._run_context, "attachments", ())
        for client in tuple(self._browser_clients.values()):
            if attachments:
                end_attachments = getattr(client, "end_attachment_session", None)
                if callable(end_attachments):
                    try:
                        end_attachments()
                    except Exception as exc:
                        debug.log(
                            "ModelClientFactory",
                            f"ATTACHMENT CLEANUP SKIP → {type(exc).__name__}: {exc}",
                        )
            bind = getattr(client, "bind_cancellation_event", None)
            if callable(bind):
                bind(None)
            elif getattr(client, "cancellation_event", None) is cancellation_event:
                client.cancellation_event = None
        if getattr(self._run_context, "cancellation_event", None) is cancellation_event:
            del self._run_context.cancellation_event
        if hasattr(self._run_context, "attachments"):
            del self._run_context.attachments

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

            models_config = self.config.get("models", {})
            model_config = (
                models_config.get(model.name, {})
                if isinstance(models_config, dict)
                else {}
            )
            if not isinstance(model_config, dict):
                model_config = {}
            pricing = model_config.get("pricing_usd_per_million", {})
            if not isinstance(pricing, dict):
                pricing = {}
            return OpenAICompatibleClient(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=model.model,
                timeout=int(provider.get("timeout", 120)),
                headers=provider.get("headers", {}),
                provider_name=model.provider,
                usage_recorder=self.token_usage_recorder,
                pricing_usd_per_million=pricing,
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
            bind = getattr(cached, "bind_cancellation_event", None)
            if callable(bind):
                bind(cancellation_event)
            else:
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

        browser_automation = self.config.get("browser_automation", {})
        if not isinstance(browser_automation, dict):
            browser_automation = {}
        provider_automation = provider.get("browser_automation", {})
        if not isinstance(provider_automation, dict):
            provider_automation = {}
        global_timing = browser_automation.get("timing", {})
        if not isinstance(global_timing, dict):
            global_timing = {}
        provider_timing = provider_automation.get("timing", {})
        if not isinstance(provider_timing, dict):
            provider_timing = {}
        global_ui = browser_automation.get("ui", {})
        if not isinstance(global_ui, dict):
            global_ui = {}
        provider_ui = provider_automation.get("ui", {})
        if not isinstance(provider_ui, dict):
            provider_ui = {}
        global_environment = browser_automation.get("environment", {})
        if not isinstance(global_environment, dict):
            global_environment = {}
        provider_environment = provider_automation.get("environment", {})
        if not isinstance(provider_environment, dict):
            provider_environment = {}
        timing_config = deep_merge_config(global_timing, provider_timing)
        ui_config = deep_merge_config(global_ui, provider_ui)
        environment_config = deep_merge_config(
            global_environment, provider_environment
        )

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
            timing_config=timing_config,
            ui_config=ui_config,
            environment_config=environment_config,
            debug_mode=False,
        )
        if cancellation_event is not None:
            client.bind_cancellation_event(cancellation_event)
        attachments = getattr(self._run_context, "attachments", ())
        if attachments:
            begin_attachments = getattr(client, "begin_attachment_session", None)
            if not callable(begin_attachments):
                client.close()
                raise RuntimeError(
                    "当前浏览器模型不支持直接上传附件；未将文件内容转换成提示词。"
                )
            try:
                begin_attachments(attachments)
            except Exception:
                client.close()
                raise
        self._browser_clients[model.name] = client
        return client
