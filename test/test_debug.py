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
    output = capsys.readouterr().out
    tracer.clear_thread_binding()

    assert "solve this file" in output
    assert "private.pdf" not in output
    assert "private.pdf" not in received[0][1]
    assert "attachment metadata redacted" in output
