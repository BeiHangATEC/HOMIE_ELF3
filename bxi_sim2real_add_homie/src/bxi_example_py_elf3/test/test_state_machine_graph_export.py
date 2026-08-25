from __future__ import annotations

from pathlib import Path

from bxi_example_py_elf3.framework.runtime.state_machine import RobotStateMachine


class _CaptureLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_graph_export_permission_failure_is_nonfatal(tmp_path: Path) -> None:
    logger = _CaptureLogger()
    machine = object.__new__(RobotStateMachine)
    machine._logger = logger

    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    target = parent_file / "state-machine.dot"

    machine._write_graph_file(str(target), "digraph test {}\n")

    assert logger.warnings
    assert str(target) in logger.warnings[0]
    assert "graph export skipped" in logger.warnings[0]
