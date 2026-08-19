import pytest

from opencompass.runners.base import BaseRunner


class StubRunner(BaseRunner):

    def __init__(self, status):
        super().__init__(task=dict(type='stub'))
        self.status = status

    def launch(self, tasks):
        return iter(self.status)


def test_runner_returns_success_statuses():
    runner = StubRunner([('first', 0), ('second', 0)])

    assert runner([]) == [('first', 0), ('second', 0)]


def test_runner_raises_when_a_child_task_fails():
    runner = StubRunner([('good', 0), ('bad', 3)])

    with pytest.raises(RuntimeError,
                       match=r'Runner tasks failed: bad \(code 3\)'):
        runner([])
