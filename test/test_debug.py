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
