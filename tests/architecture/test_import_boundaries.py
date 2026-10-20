"""Static architecture checks that run in CI without importing application code."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "study_agent"


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(("study_agent", *parts))


def _resolve_import(node: ast.Import | ast.ImportFrom, package: str) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if node.level:
        relative = "." * node.level + (node.module or "")
        return {importlib.util.resolve_name(relative, package)}
    return {node.module} if node.module else set()


def _imports(path: Path) -> set[str]:
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            result.update(_resolve_import(node, package))
    return result


def _layer_for(module: str) -> str | None:
    for layer in ("domain", "ports", "application", "infrastructure"):
        if module == f"study_agent.{layer}" or module.startswith(f"study_agent.{layer}."):
            return layer
    return None


def _allowed_first_party(layer: str) -> tuple[str, ...]:
    if layer == "domain":
        return ("study_agent.domain",)
    if layer == "ports":
        return ("study_agent.domain", "study_agent.ports")
    if layer == "application":
        return (
            "study_agent.domain",
            "study_agent.ports",
            "study_agent.application",
        )
    if layer == "infrastructure":
        return (
            "study_agent.domain",
            "study_agent.ports",
            "study_agent.infrastructure",
        )
    return ()


def test_new_architecture_layers_use_only_declared_dependencies() -> None:
    assert PACKAGE_ROOT.exists(), "new architecture package was not created"
    python_files = sorted(PACKAGE_ROOT.rglob("*.py"))
    violations: list[str] = []

    for path in python_files:
        module = _module_name(path)
        layer = _layer_for(module)
        if layer is None:
            # Package initializers and the composition root are intentionally
            # allowed to expose packages / wire concrete implementations.
            if module in {"study_agent", "study_agent.bootstrap"}:
                continue
            violations.append(f"{path.relative_to(ROOT)} is outside a declared layer")
            continue

        for imported in sorted(_imports(path)):
            top_level = imported.split(".", 1)[0]
            if top_level == "study_agent":
                if not any(
                    imported == prefix or imported.startswith(prefix + ".")
                    for prefix in _allowed_first_party(layer)
                ):
                    violations.append(f"{module} ({layer}) imports forbidden layer/module {imported}")
            elif top_level not in sys.stdlib_module_names:
                if layer == "infrastructure" and imported == "core.storage_security":
                    # Transitional adapter: reuse the existing hardened path
                    # guard until the storage security module itself is migrated.
                    continue
                violations.append(f"{module} ({layer}) imports non-standard dependency {imported}")

    assert not violations, "Architecture dependency violations:\n" + "\n".join(violations)


def test_domain_has_no_infrastructure_or_legacy_core_imports() -> None:
    domain_root = PACKAGE_ROOT / "domain"
    forbidden_prefixes = ("core", "config", "tools", "coder", "web", "integrations")
    violations = []
    for path in domain_root.rglob("*.py"):
        module = _module_name(path)
        for imported in _imports(path):
            top_level = imported.split(".", 1)[0]
            if top_level in forbidden_prefixes:
                violations.append(f"{module} imports {imported}")
    assert not violations, "Domain must stay framework-independent:\n" + "\n".join(violations)


def test_pytest_configuration_includes_new_src_package() -> None:
    config = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "pythonpath = . src" in config
    assert "testpaths = test tests" in config
