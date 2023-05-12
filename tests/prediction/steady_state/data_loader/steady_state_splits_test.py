from simba_ml.prediction.steady_state.data_loader import splits
from tests.prediction.help_functions import create_data


def test_train_test_split():
    data = create_data()
    test_split = 0.2
    train, test = splits.train_test_split(data=data, test_split=test_split)
    assert len(train) == len(data) * (1 - test_split)
    assert len(test) == len(data) * test_split
