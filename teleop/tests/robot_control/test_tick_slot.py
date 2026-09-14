"""Unit tests for LatestTickSlot: always offer, take once, overwrite pending."""
import threading
import time

from teleop.robot_control.tick_slot import LatestTickSlot, now_us


def test_offer_then_take_once():
    slot = LatestTickSlot()
    assert slot.offer(1001) is True
    assert slot.pending() is True
    assert slot.take() == 1001
    assert slot.take() is None
    assert slot.pending() is False


def test_offer_overwrites_pending():
    slot = LatestTickSlot()
    assert slot.offer(10) is True
    assert slot.offer(11) is False
    assert slot.take() == 11
    assert slot.offer(12) is True
    assert slot.take() == 12


def test_consumed_ts_not_reused():
    slot = LatestTickSlot()
    slot.offer(42)
    first = slot.take()
    second = slot.take()
    assert first == 42
    assert second is None


def test_wait_take_unblocks_on_offer():
    slot = LatestTickSlot()
    got = []

    def waiter():
        got.append(slot.wait_take(timeout=1.0))

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.02)
    slot.offer(99)
    thread.join(timeout=1.0)
    assert got == [99]


def test_wait_take_timeout_returns_none():
    slot = LatestTickSlot()
    assert slot.wait_take(timeout=0.02) is None


def test_now_us_is_int_and_advances():
    a = now_us()
    b = now_us()
    assert isinstance(a, int)
    assert b >= a
