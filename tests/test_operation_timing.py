import pytest

from core.operation_timing import OperationTiming


def midpoint(low, high):
    return (low + high) / 2


def test_operation_timing_samples_normal_send_interval_and_fixed_retry_wait():
    timing = OperationTiming(random_uniform=midpoint)

    assert timing.sample("send_interval_seconds") == pytest.approx(10.5)
    assert timing.sample("retry_wait_seconds") == 20.0
    assert 6.0 <= timing.sample("send_interval_seconds") <= 15.0


def test_operation_timing_uses_overrides_and_preserves_default_ui_ranges():
    timing = OperationTiming(
        {
            "send_interval_seconds": {"min": 7, "max": 9},
            "retry_wait_seconds": 20,
            "ui_delay_seconds": {"click": {"min": 0.2, "max": 0.4}},
        },
        random_uniform=midpoint,
    )

    assert timing.sample("send_interval_seconds") == pytest.approx(8.0)
    assert timing.sample("click") == pytest.approx(0.3)
    # Unspecified UI delay keys still fall back to config/browser_automation.json.
    assert timing.sample("upload_poll") >= 0.5
    assert timing.sample("upload_poll") <= 1.0


def test_operation_timing_wait_is_cancellable_and_uses_injected_sleep():
    timing = OperationTiming(
        {"ui_delay_seconds": {"click": {"min": 0.1, "max": 0.3}}},
        random_uniform=midpoint,
    )
    events = []

    elapsed = timing.wait(
        "click",
        sleep=lambda seconds: events.append(("sleep", seconds)),
        check_cancelled=lambda: events.append(("check", None)),
    )

    assert elapsed == pytest.approx(0.2)
    assert events == [
        ("check", None),
        ("sleep", pytest.approx(0.2)),
        ("check", None),
    ]


@pytest.mark.parametrize(
    "config",
    [
        {"send_interval_seconds": {"min": 15, "max": 6}},
        {"retry_wait_seconds": -1},
        {"ui_delay_seconds": {"click": {"min": 0.5, "max": 0.1}}},
    ],
)
def test_operation_timing_rejects_invalid_ranges(config):
    with pytest.raises(ValueError):
        OperationTiming(config)


def test_operation_timing_legacy_fixed_interval_remains_available():
    timing = OperationTiming(
        fixed_send_interval_seconds=5.0,
        random_uniform=midpoint,
    )

    assert timing.sample("send_interval_seconds") == 5.0
