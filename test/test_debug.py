from core.__debug__ import DebugTracer


def test_debug_tracer_emits_when_enabled(capsys):
    tracer = DebugTracer(enabled=True)
    tracer.log("Test", "hello")
    output = capsys.readouterr().out

    assert "[DEBUG " in output
    assert "[Test] hello" in output


def test_debug_tracer_is_silent_when_disabled(capsys):
    tracer = DebugTracer(enabled=False)
    tracer.log("Test", "hello")
    assert capsys.readouterr().out == ""



def test_debug_tracer_prints_to_terminal_and_calls_sink_when_enabled(capsys):
    tracer = DebugTracer(enabled=True)
    received = []
    tracer.bind_thread(lambda module, message: received.append((module, message)))

    tracer.log("BrowserModel", "WINDOW BACK ATTEMPT")

    output = capsys.readouterr().out
    tracer.clear_thread_binding()
    assert "[BrowserModel] WINDOW BACK ATTEMPT" in output
    assert received == [("BrowserModel", "WINDOW BACK ATTEMPT")]


def test_debug_tracer_keeps_terminal_quiet_when_disabled_even_with_sink(capsys):
    tracer = DebugTracer(enabled=False)
    received = []
    tracer.bind_thread(lambda module, message: received.append((module, message)))

    tracer.log("BrowserModel", "WINDOW BACK ATTEMPT")

    assert capsys.readouterr().out == ""
    assert received == [("BrowserModel", "WINDOW BACK ATTEMPT")]
    tracer.clear_thread_binding()



def test_debug_tracer_redacts_attachment_metadata_from_terminal_and_sink(capsys):
    tracer = DebugTracer(enabled=True)
    received = []
    tracer.bind_thread(lambda module, message: received.append((module, message)))
    tracer.log(
        "StudyAgent",
        'QUESTION → solve this file\\n【原始文件附件】\\n文件名（JSON）：["private.pdf"]',
    )
    tracer.log(
        "OtherModule",
        'CONTEXT → query="private request" 【原始文件附件】文件名（JSON）：["private.pdf"]',
    )
    output = capsys.readouterr().out
    tracer.clear_thread_binding()

    assert "solve this file" not in output
    assert "private.pdf" not in output
    assert "private.pdf" not in received[0][1]
    assert "user request redacted" in output
    assert "attachment metadata redacted" in output



def test_debug_tracer_redacts_goal_and_knowledge_graph_query_text(capsys):
    tracer = DebugTracer(enabled=True)
    received = []
    tracer.bind_thread(lambda module, message: received.append((module, message)))

    tracer.log(
        "GoalMatcher",
        "MATCH → current='private course question', candidates=2, matched=0",
    )
    tracer.log(
        "KnowledgeGraph",
        "RESOLVE → query='private homework prompt', matched=[]",
    )
    tracer.log(
        "KnowledgeGraph",
        "CONTEXT → query='private homework prompt', matched=[], neighbors=0",
    )

    output = capsys.readouterr().out
    tracer.clear_thread_binding()
    for sensitive in ("private course question", "private homework prompt"):
        assert sensitive not in output
        assert all(sensitive not in message for _module, message in received)
    assert "current=[redacted; chars=" in output
    assert "query=[redacted; chars=" in output



def test_debug_tracer_redacts_calculation_and_verification_text(capsys):
    tracer = DebugTracer(enabled=True)
    received = []
    tracer.bind_thread(lambda module, message: received.append((module, message)))

    tracer.log(
        "ToolExecutor",
        "CALCULATE → expression='private salary * bonus', result_type=float, result_size≈8",
    )
    tracer.log(
        "ToolExecutor",
        "VERIFY → claim='private medical diagnosis', evidence_count=0, status=UNCERTAIN",
    )
    tracer.log(
        "AgentToolLoop",
        "VERIFY RESULT → claim='private account number', status=PASS",
    )

    output = capsys.readouterr().out
    tracer.clear_thread_binding()
    for sensitive in ("private salary * bonus", "private medical diagnosis", "private account number"):
        assert sensitive not in output
        assert all(sensitive not in message for _module, message in received)
    assert "result_type=float" in output
    assert "status=UNCERTAIN" in output


def test_debug_tracer_redacts_credentials_and_host_paths_from_all_sinks(capsys):
    tracer = DebugTracer(enabled=True)
    received = []
    tracer.bind_thread(lambda module, message: received.append((module, message)))

    tracer.log(
        "Provider",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345 "
        "api_key=do-not-log-this C:\\Users\\Alice\\private.txt",
    )

    output = capsys.readouterr().out
    tracer.clear_thread_binding()
    combined = output + repr(received)
    for secret in (
        "abcdefghijklmnopqrstuvwxyz012345",
        "do-not-log-this",
        "C:\\Users\\Alice\\private.txt",
    ):
        assert secret not in combined
    assert "[REDACTED" in combined
