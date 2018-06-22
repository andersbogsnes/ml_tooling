from ml_utils.transformers import (Select,
                                    FillNA,
                                    ToCategorical,
                                    FuncTransformer,
                                    Binner,
                                    Renamer,
                                    TransformerError,
                                    DateEncoder,
                                    FreqFeature)
import pytest
import pandas as pd
import numpy as np

np.random.seed(42)


def test_df_selector_returns_correct_dataframe(categorical):
    select = Select(['a'])
    result = select.fit_transform(categorical)

    assert isinstance(result, pd.DataFrame)
    assert len(categorical) == len(result)
    assert {'a'} == set(result.columns)


def test_df_selector_with_nonlist(categorical):
    select = Select('a')
    result = select.fit_transform(categorical)
    assert isinstance(result, pd.DataFrame)
    assert len(categorical) == len(result)
    assert {'a'} == set(result.columns)


def test_df_selector_with_multiple_columns(categorical):
    select = Select(['a', 'b'])
    result = select.fit_transform(categorical)

    assert isinstance(result, pd.DataFrame)
    assert len(categorical) == len(result)
    assert {'a', 'b'} == set(result.columns)


def test_imputer_returns_correct_dataframe(categorical_na):
    imputer = FillNA('Unknown')
    result = imputer.fit_transform(categorical_na)

    assert isinstance(result, pd.DataFrame)
    assert len(categorical_na) == len(result)
    assert {'a', 'b'} == set(result.columns)
    assert 'Unknown' == result.iloc[0, 1]
    assert 'Unknown' == result.iloc[1, 0]


def test_imputer_returns_dataframe_unchanged_if_no_nans(categorical):
    imputer = FillNA('Unknown')
    result = imputer.fit_transform(categorical)

    assert isinstance(result, pd.DataFrame)
    assert len(categorical) == len(result)
    assert {'a', 'b'} == set(result.columns)
    assert ~result.isin(['Unknown']).any().any()


def test_to_categorical_returns_correct_dataframe(categorical):
    to_cat = ToCategorical()
    result = to_cat.fit_transform(categorical)
    expected_cols = ['a_a1', 'a_a2', 'a_a3', 'b_b1', 'b_b2', 'b_b3']

    assert isinstance(result, pd.DataFrame)
    assert len(categorical) == len(result)
    assert set(expected_cols) == set(result.columns)
    for col in expected_cols:
        assert pd.api.types.is_numeric_dtype(result[col])


def test_to_categorical_discards_unseen_values(categorical):
    to_cat = ToCategorical()
    to_cat.fit(categorical)
    new_data = pd.DataFrame({"a": ["a1", "a2", "ab1"],
                             "b": ["b1", "b2", "ab2"]})

    result = to_cat.transform(new_data)
    expected_cols = ['a_a1', 'a_a2', 'a_a3', 'b_b1', 'b_b2', 'b_b3']

    assert isinstance(result, pd.DataFrame)
    assert 0 == result.isna().sum().sum()
    assert set(expected_cols) == set(result.columns)


def test_func_transformer_returns_correctly(categorical):
    func_transformer = FuncTransformer(lambda x: x.str.upper())
    result = func_transformer.fit_transform(categorical)

    assert isinstance(result, pd.DataFrame)
    assert len(categorical) == len(result)
    for col in result.columns:
        assert result[col].str.isupper().all()


def test_binner_returns_correctly(numerical):
    labels = ['1', '2', '3', '4']
    binner = Binner(bins=[0, 1, 2, 3, 4], labels=labels)
    result = binner.fit_transform(numerical[['a']])

    assert isinstance(result, pd.DataFrame)
    assert len(numerical) == len(result)
    assert pd.api.types.is_categorical_dtype(result['a'])
    assert 4 == len(result['a'].cat.categories)
    assert set(labels) == set(result['a'].cat.categories)


def test_binner_returns_nan_on_unseen_data(numerical):
    labels = ['1', '2', '3', '4']
    binner = Binner(bins=[0, 1, 2, 3, 4], labels=labels)
    binner.fit(numerical[['a']])

    new_data = pd.DataFrame({"a": [5, 6, 7, 8]})
    result = binner.transform(new_data)

    assert isinstance(result, pd.DataFrame)
    assert len(new_data) == len(result)
    assert len(new_data) == result.isna().sum().sum()


def test_renamer_returns_correctly(numerical):
    new_col_names = ['test_a', 'test_b']
    renamer = Renamer(new_col_names)
    result = renamer.fit_transform(numerical)

    assert isinstance(result, pd.DataFrame)
    assert len(numerical) == len(result)
    assert set(new_col_names) == set(result.columns)


def test_mismatched_no_of_names_raises(numerical):
    new_col_names = ['test_a']
    renamer = Renamer(new_col_names)
    with pytest.raises(TransformerError):
        renamer.fit_transform(numerical)


def test_date_encoder_returns_correctly(dates):
    date_coder = DateEncoder()
    result = date_coder.fit_transform(dates)

    assert isinstance(result, pd.DataFrame)
    assert 4 == len(result.columns)
    assert len(dates) == len(result)
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col])
    assert 'a' not in result.columns


def test_freqfeature_returns_correctly(categorical):
    freq_feature = FreqFeature()
    result = freq_feature.fit_transform(categorical)

    assert isinstance(result, pd.DataFrame)
    assert len(categorical) == len(result)
    assert set(categorical.columns) == set(result.columns)
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col])
    assert 1 / len(categorical) == result.iloc[0, 0]
    assert all(1 == result.sum())


def test_freqfeature_handles_nans_correctly(categorical_na):
    freq_feature = FreqFeature()
    result = freq_feature.fit_transform(categorical_na)

    assert isinstance(result, pd.DataFrame)
    assert len(categorical_na) == len(result)
    assert set(categorical_na.columns) == set(result)
    assert 0 == result.isna().sum().sum()
    assert 1 / (len(categorical_na) - 1) == result.iloc[0, 0]


def test_freq_features_returns_0_when_unseen_value_is_given(categorical):
    freq_feature = FreqFeature()
    freq_feature.fit(categorical)

    new_data = pd.DataFrame({"a": ["a1", "a2", "c25"]})
    result = freq_feature.transform(new_data)

    assert isinstance(result, pd.DataFrame)
    assert len(new_data) == len(result)
    assert 0 == result.iloc[-1, 0]
