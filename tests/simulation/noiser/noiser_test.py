import pandas as pd

from simba_ml.simulation import noisers
from simba_ml.simulation import distributions


def test_no_noiser():
    """Checks that the `NoNoiser` does not put any noise to the incoming signal."""
    before = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]})
    after = noisers.no_noiser.NoNoiser().noisify(before)
    assert before.equals(after)


def test_addititive_noiser():
    """Checks, that the AdditiveNoiser puts Noise to the incoming signal."""
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    additional_noiser = noisers.additive_noiser.AdditiveNoiser(
        distributions.NormalDistribution(1, 0)
    )
    expected = pd.DataFrame({"A": [2.0, 3.0, 4.0], "B": [3.0, 4.0, 5.0]})
    assert expected.equals(additional_noiser.noisify(before))


def test_multiplicative_noiser():
    """Checks, that the `MultiplicativeNoiser` puts Noise to the incoming signal."""
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    additional_noiser = noisers.multiplicative_noiser.MultiplicativeNoiser(
        distributions.NormalDistribution(2, 0)
    )
    expected = pd.DataFrame({"A": [2.0, 4.0, 6.0], "B": [4.0, 6.0, 8.0]})
    assert expected.equals(additional_noiser.noisify(before))


def test_multi_noiser():
    """Checks, that the `MultiNoiser` puts Noise to the incoming signal
    by appliyng one of the given `Noiser`."""
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    additional_noiser = noisers.additive_noiser.AdditiveNoiser(
        distributions.NormalDistribution(1, 0)
    )
    multiplicative_noiser = noisers.multiplicative_noiser.MultiplicativeNoiser(
        distributions.NormalDistribution(2, 0)
    )
    multi_noiser = noisers.multi_noiser.MultiNoiser(
        [additional_noiser, multiplicative_noiser]
    )
    expected1 = pd.DataFrame({"A": [2.0, 3.0, 4.0], "B": [3.0, 4.0, 5.0]})
    expected2 = pd.DataFrame({"A": [2.0, 4.0, 6.0], "B": [4.0, 6.0, 8.0]})
    result = multi_noiser.noisify(before)
    assert expected1.equals(result) or expected2.equals(result)


def test_sequential_noiser():
    """Checks, that the `MultiNoiser` puts Noise to the incoming signal
    by appliyng the given `Noiser` sequentially."""
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    additional_noiser = noisers.additive_noiser.AdditiveNoiser(
        distributions.NormalDistribution(1, 0)
    )
    multiplicative_noiser = noisers.multiplicative_noiser.MultiplicativeNoiser(
        distributions.NormalDistribution(2, 0)
    )
    sequential_noiser = noisers.sequential_noiser.SequentialNoiser(
        [additional_noiser, multiplicative_noiser]
    )
    expected = pd.DataFrame({"A": [4.0, 6.0, 8.0], "B": [6.0, 8.0, 10.0]})
    assert expected.equals(sequential_noiser.noisify(before))


def test_adjusting_mean_noiser():
    """Checks that the `AdjustingMeanNoiser` puts noise to the incoming signal
    approaching the mean."""
    before = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]})
    expected = pd.DataFrame({"A": [2, 2.5, 3, 3.5, 4], "B": [3, 3.5, 4, 4.5, 5]})

    after = noisers.adjusting_mean_noiser.AdjustingMeanNoiser(
        weight=distributions.Constant(0.5)
    ).noisify(before)
    assert expected.equals(after)
    after = noisers.adjusting_mean_noiser.AdjustingMeanNoiser(
        weight=distributions.Constant(0)
    ).noisify(before)
    expected = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]})
    assert expected.equals(after)


def test_elastic_noiser():
    """Checks, that the `ElasticNoiser` puts Noise and ouptput =/= input."""
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    not_expected = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    after = noisers.elastic_noiser.ElasticNoiser(k=distributions.Constant(10)).noisify(
        before
    )
    assert not not_expected.equals(after)
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    after = noisers.elastic_noiser.ElasticNoiser(
        k=distributions.Constant(10), exponential=True
    ).noisify(before)
    assert not not_expected.equals(after)


def test_elastic_inverted():
    """Checks, that the `ElasticNoiser` puts Noise and ouptput =/= input."""
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    not_expected = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    after = noisers.elastic_noiser.ElasticNoiser(
        k=distributions.Constant(10), invert=True
    ).noisify(before)
    assert not not_expected.equals(after)
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    after = noisers.elastic_noiser.ElasticNoiser(
        k=distributions.Constant(10), exponential=True, invert=True
    ).noisify(before)
    assert not not_expected.equals(after)


def test_column_noiser():
    """Checks, that the `ColumnNoiser` puts noise on columns individually."""
    before = pd.DataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [10, 11, 12]}
    )
    expected = pd.DataFrame(
        {"A": [3, 4, 5], "B": [8, 10, 12], "C": [7, 8, 9], "D": [10, 11, 12]}
    )
    col_noisers = {
        "A": noisers.additive_noiser.AdditiveNoiser(distributions.Constant(2)),
        "B": noisers.multiplicative_noiser.MultiplicativeNoiser(
            distributions.Constant(2)
        ),
        "D": noisers.no_noiser.NoNoiser(),
    }
    column_noiser = noisers.column_noiser.ColumnNoiser(col_noisers)
    after = column_noiser.noisify(before)
    assert after.equals(expected)
