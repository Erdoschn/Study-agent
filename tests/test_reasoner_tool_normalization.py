from core.reasoner import AgentReasoner


def test_reasoner_normalizes_tool_object_name():
    decision = AgentReasoner._parse(
        '{"action":"ASSESS","tool":{"name":"assess"},'
        '"reasoning_summary":"assess the evidence","arguments":{}}'
    )

    assert decision.action == "ASSESS"
    assert decision.tool == "assess"


def test_reasoner_still_rejects_mismatched_tool_object_name():
    try:
        AgentReasoner._parse(
            '{"action":"ASSESS","tool":{"name":"search"},'
            '"reasoning_summary":"wrong tool","arguments":{}}'
        )
    except RuntimeError as exc:
        assert "应为 assess" in str(exc)
    else:
        raise AssertionError("Expected mismatched action/tool to be rejected")
