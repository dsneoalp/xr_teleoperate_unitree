"""Unit tests for OneShotTickSlot: one tick, consume once, never reuse."""
from teleop.robot_control.tick_slot import OneShotTickSlot, now_us


def test_try_publish_then_take_once():
    slot = OneShotTickSlot()
    assert slot.try_publish(1001) is True
    assert slot.pending() is True
    assert slot.take() == 1001
    assert slot.take() is None
    assert slot.pending() is False


def test_second_publish_rejected_while_pending():
    slot = OneShotTickSlot()
    assert slot.try_publish(10) is True
    assert slot.try_publish(11) is False
    assert slot.take() == 10
    assert slot.try_publish(11) is True
    assert slot.take() == 11


def test_consumed_ts_not_reused():
    slot = OneShotTickSlot()
    slot.try_publish(42)
    first = slot.take()
    second = slot.take()
    assert first == 42
    assert second is None


def test_now_us_is_int_and_advances():
    a = now_us()
    b = now_us()
    assert isinstance(a, int)
    assert b >= a
