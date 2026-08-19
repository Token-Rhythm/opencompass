import pytest
from datasets import Dataset

from opencompass.openicl.icl_dataset_reader import load_partial_dataset


def test_load_partial_dataset_accepts_explicit_indices():
    dataset = Dataset.from_dict({'value': list(range(8))})

    selected = load_partial_dataset(dataset, [1, 4, 7])

    assert selected['value'] == [1, 4, 7]


@pytest.mark.parametrize('indices', [[-1], [8], [1.5]])
def test_load_partial_dataset_rejects_invalid_explicit_indices(indices):
    dataset = Dataset.from_dict({'value': list(range(8))})

    with pytest.raises(ValueError, match='Explicit dataset indices'):
        load_partial_dataset(dataset, indices)
