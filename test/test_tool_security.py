import pytest

from core.tool_loop import ToolExecutor, ToolSpec


def test_agent_tool_allowlist_contains_only_safe_learning_tools():
    executor = ToolExecutor(search_router=None)

    assert set(executor.tool_specs()[i]["name"] for i in range(len(executor.tool_specs()))) == {
        "search",
        "calculate",
        "assess",
        "verify",
    }


@pytest.mark.parametrize(
    "name",
    [
        "delete_file",
        "remove_file",
        "write_file",
        "shell",
        "execute_code",
        "run_code",
        "python",
        "python_exec",
        "bash",
        "powershell",
        "run_command",
    ],
)
def test_forbidden_tools_cannot_be_registered(name):
    executor = ToolExecutor(search_router=None)
    with pytest.raises(PermissionError):
        executor.register(ToolSpec(name, "dangerous", {"type": "object"}, lambda *_args: None))


@pytest.mark.parametrize(
    "name",
    [
        "delete_file",
        "shell",
        "execute_code",
        "python_exec",
        "run_command",
    ],
)
def test_forbidden_tools_cannot_be_invoked(name):
    executor = ToolExecutor(search_router=None)
    with pytest.raises(ValueError):
        executor.execute(name, {})


def test_unknown_tool_is_rejected_even_when_name_is_similar():
    executor = ToolExecutor(search_router=None)
    with pytest.raises(ValueError):
        executor.execute("search_file", {"query": "x"})


def test_calculate_is_not_general_code_execution():
    executor = ToolExecutor(search_router=None)

    with pytest.raises(ValueError):
        executor.execute("calculate", {"expression": "__import__('os').system('echo unsafe')"})

    with pytest.raises(ValueError):
        executor.execute("calculate", {"expression": "[1, 2, 3]"})
