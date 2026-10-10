from coder import agent as agent_module


def test_coder_agent_initializes_debug_mode_before_reasoner(tmp_path, monkeypatch):
    seen = []

    class FakeReasoner:
        def __init__(
            self,
            *,
            debug_mode=False,
            reuse_chat=True,
            min_send_interval_seconds=5.0,
        ):
            seen.append((debug_mode, reuse_chat, min_send_interval_seconds))

    monkeypatch.setattr(agent_module, "CoderReasoner", FakeReasoner)

    agent_module.CoderAgent(
        workspace=tmp_path,
        search_router=object(),
        harness=object(),
        debug_mode=True,
    )

    assert seen == [(True, True, 5.0)]


def test_coder_agent_defaults_debug_mode_to_false(tmp_path, monkeypatch):
    seen = []

    class FakeReasoner:
        def __init__(
            self,
            *,
            debug_mode=False,
            reuse_chat=True,
            min_send_interval_seconds=5.0,
        ):
            seen.append((debug_mode, reuse_chat, min_send_interval_seconds))

    monkeypatch.setattr(agent_module, "CoderReasoner", FakeReasoner)

    agent_module.CoderAgent(
        workspace=tmp_path,
        search_router=object(),
        harness=object(),
    )

    assert seen == [(False, True, 5.0)]


def test_coder_agent_handles_model_selected_new_chat(tmp_path):
    actions = iter([
        {
            "action": "PLAN",
            "arguments": {},
            "goal": {
                "description": "修复一个小 bug",
                "scope_files": ["a.py"],
                "milestones": ["inspect", "fix", "verify"],
                "success_criteria": ["bug is fixed"],
                "must_create_tests": False,
                "must_pass_tests": False,
            },
        },
        {"action": "NEW_CHAT", "arguments": {}},
        {"action": "FINISH", "arguments": {}},
    ])
    new_chat_calls = []

    class FakeReasoner:
        def decide(self, state, tool_specs):
            return next(actions)

        def new_chat(self):
            new_chat_calls.append(True)

    class FakeHarness:
        sandbox = object()
        backup = object()

        def tool_specs(self):
            return []

        def execute(self, action, arguments, state):
            assert action == "VERIFY_GOAL"
            return {"verified": True}

    agent = agent_module.CoderAgent(
        workspace=tmp_path,
        reasoner=FakeReasoner(),
        harness=FakeHarness(),
        max_runtime_seconds=30,
    )
    state = agent.run("修复一个小 bug")

    assert state.finished is True
    assert state.project == tmp_path.name
    assert state.goal_verified is True
    assert state.chat_resets == 1
    assert state.metrics["chat_resets"] == 1
    assert new_chat_calls == [True]
    assert [step.action for step in state.steps] == ["PLAN", "NEW_CHAT", "FINISH"]


def test_coder_reasoner_accepts_patch_notebook_action():
    from coder.reasoner import CoderReasoner

    decision = CoderReasoner._parse(
        '{"action":"PATCH_NOTEBOOK","arguments":{"path":"demo.ipynb","cell_index":0,"old_source":"x","new_source":"y"}}'
    )

    assert decision["action"] == "PATCH_NOTEBOOK"
    assert decision["arguments"]["cell_index"] == 0


def test_coder_reasoner_accepts_new_chat_action():
    from coder.reasoner import CoderReasoner

    decision = CoderReasoner._parse(
        '{"action":"NEW_CHAT","arguments":{},"reasoning_summary":"上下文需要重置"}'
    )

    assert decision["action"] == "NEW_CHAT"
    assert decision["arguments"] == {}


def test_coder_reasoner_new_chat_delegates_to_browser_model():
    from coder.reasoner import CoderReasoner

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def new_chat(self):
            self.calls += 1

    model = FakeModel()
    reasoner = CoderReasoner(model=model)

    reasoner.new_chat()

    assert model.calls == 1


def test_coder_reasoner_defaults_to_reused_chat():
    from coder.reasoner import CoderReasoner

    class FakeModel:
        reuse_chat = True

    reasoner = CoderReasoner(model=FakeModel())

    assert reasoner.model.reuse_chat is True


def test_coder_reasoner_accepts_configurable_chat_policy(monkeypatch):
    from coder.reasoner import CoderReasoner

    seen = {}

    class FakeBrowserModel:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("coder.reasoner.BrowserModel", FakeBrowserModel)
    reasoner = CoderReasoner(
        reuse_chat=False,
        min_send_interval_seconds=9.0,
    )

    assert reasoner.model is not None
    assert seen["reuse_chat"] is False
    assert seen["min_send_interval_seconds"] == 9.0



def test_coder_agent_stops_when_cancellation_event_is_set(tmp_path):
    import threading

    calls = []

    class FakeReasoner:
        model = None

        def decide(self, state, tools):
            calls.append(state.step_count)
            event.set()
            return {"action": "FINISH", "arguments": {}}

    class FakeHarness:
        class Sandbox:
            def preflight(self):
                return None

        sandbox = Sandbox()

        class Backup:
            def ensure_initial_snapshot(self, generation):
                return type("Snapshot", (), {"generation": generation})()

        backup = Backup()

        def tool_specs(self):
            return []

    event = threading.Event()
    agent = agent_module.CoderAgent(
        workspace=tmp_path,
        reasoner=FakeReasoner(),
        harness=FakeHarness(),
        max_runtime_seconds=30,
        cancellation_event=event,
    )

    state = agent.run("cancel me")

    assert state.cancelled is True
    assert state.finished is True
    assert state.goal_verified is False
    assert state.error is None
    assert state.summary == "任务已被用户中止。"
    assert calls == [0]



def _stop_test_agent(tmp_path, actions):
    class FakeReasoner:
        model = None

        def decide(self, state, tools):
            return next(actions)

    class FakeHarness:
        sandbox = object()
        backup = object()

        def tool_specs(self):
            return []

        def execute(self, action, arguments, state):
            assert action == "VERIFY_GOAL"
            return {"verified": True}

    return agent_module.CoderAgent(
        workspace=tmp_path,
        reasoner=FakeReasoner(),
        harness=FakeHarness(),
        max_runtime_seconds=30,
    )


def _stop_test_plan():
    return {
        "action": "PLAN",
        "arguments": {},
        "goal": {
            "description": "处理当前任务",
            "scope_files": ["task.py"],
            "milestones": ["inspect", "implement", "verify"],
            "success_criteria": ["task completed"],
            "must_create_tests": False,
            "must_pass_tests": False,
        },
    }


def test_coder_agent_requires_explicit_confirmation_before_stopping(tmp_path):
    actions = iter([
        _stop_test_plan(),
        {"action": "STOP", "arguments": {}, "reasoning_summary": "需要停止"},
    ])
    questions = []
    agent = _stop_test_agent(tmp_path, actions)

    state = agent.run(
        "完成当前任务",
        user_interaction=lambda question: questions.append(question) or "确认终止",
    )

    assert len(questions) == 1
    assert "确认终止" in questions[0]
    assert state.finished is True
    assert state.cancelled is True
    assert state.goal_verified is False
    assert state.error is None
    assert state.summary == "任务已根据用户二次确认终止。"
    assert state.steps[-1].observation["status"] == "confirmed_stopped"


def test_coder_agent_continues_when_stop_confirmation_is_declined(tmp_path):
    actions = iter([
        _stop_test_plan(),
        {"action": "STOP", "arguments": {}, "reasoning_summary": "当前无法继续"},
        # A repeated STOP must not override the user's explicit choice to continue.
        {"action": "STOP", "arguments": {}, "reasoning_summary": "重复请求停止"},
        {"action": "FINISH", "arguments": {}},
    ])
    questions = []
    agent = _stop_test_agent(tmp_path, actions)

    state = agent.run(
        "完成当前任务",
        user_interaction=lambda question: questions.append(question) or "继续任务",
    )

    assert len(questions) == 1
    assert state.finished is True
    assert state.goal_verified is True
    assert state.cancelled is False
    assert [step.observation.get("status") for step in state.steps if step.action == "STOP"] == [
        "confirmation_declined",
        "stop_blocked_after_user_declined",
    ]


def test_coder_agent_does_not_stop_on_ambiguous_or_missing_confirmation(tmp_path):
    for response in (None, "我还没想好"):
        actions = iter([
            _stop_test_plan(),
            {"action": "STOP", "arguments": {}, "reasoning_summary": "需要停止"},
            {"action": "FINISH", "arguments": {}},
        ])
        agent = _stop_test_agent(tmp_path, actions)
        state = agent.run(
            "完成当前任务",
            user_interaction=lambda _question, value=response: value,
        )
        assert state.finished is True
        assert state.goal_verified is True
        assert state.cancelled is False
        assert any(
            step.action == "STOP"
            and step.observation.get("status") == "confirmation_unanswered"
            for step in state.steps
        )



def test_stop_confirmation_accepts_clear_natural_language_and_rejects_negation():
    confirm = agent_module._is_explicit_stop_confirmation
    assert confirm("那不改了，取消任务") is True
    assert confirm("请终止任务") is True
    assert confirm("我不想取消任务") is False
    assert confirm("不要结束任务，继续") is False
    assert confirm("我还没想好") is False
