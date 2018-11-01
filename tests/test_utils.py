import pytest

from ml_tooling.utils import (get_git_hash,
                              find_model_file,
                              _is_percent,
                              MLToolingError,
                              get_scoring_func,
                              )

from sklearn.metrics.scorer import _PredictScorer


def test_get_git_hash_returns_correctly():
    git_hash = get_git_hash()
    assert isinstance(git_hash, str)
    assert 10 < len(git_hash)


def test_find_model_file_with_given_model_returns_correctly(tmpdir):
    model_folder = tmpdir.mkdir('model')
    model1 = 'TestModel1_1234.pkl'
    model1_file = model_folder.join(model1)
    model1_file.write('test')

    model2 = 'TestModel2_1234.pkl'
    model2_file = model_folder.join(model2)
    model2_file.write('test')

    result = find_model_file(model1_file)

    assert model1_file == result


def test_find_model_raise_when_no_model_found():
    with pytest.raises(MLToolingError, match="No models found - check your directory: nonsense"):
        find_model_file('nonsense')


def test_find_model_file_if_multiple_with_same_hash(tmpdir, monkeypatch):
    def mockreturn():
        return '1234'

    monkeypatch.setattr('ml_tooling.utils.get_git_hash', mockreturn)

    model_folder = tmpdir.mkdir('model')
    model1 = 'TestModel1_1234.pkl'
    model1_file = model_folder.join(model1)
    model1_file.write('test')
    first_file_mtime = model1_file.mtime()

    model2 = 'TestModel2_1234.pkl'
    model2_file = model_folder.join(model2)
    model2_file.write('test')
    model2_file.setmtime(first_file_mtime + 100)  # Ensure second file is newer

    result = find_model_file(model_folder)

    assert model2_file == result


@pytest.mark.parametrize('number, is_percent', [
    (0.2, True),
    (1, False),
    (10, False),
    (.00000000001, True),
    (1000000, False)
])
def test_is_percent_returns_correctly(number, is_percent):
    assert _is_percent(number) is is_percent


def test_is_percent_raises_correctly_if_given_large_float():
    with pytest.raises(ValueError, match='Floats only valid between 0 and 1. Got 100.0'):
        _is_percent(100.0)


def test_scoring_func_returns_a_scorer(classifier):
    scorer = get_scoring_func('accuracy')

    score = scorer(classifier.model, classifier.data.test_x, classifier.data.test_y)
    assert isinstance(scorer, _PredictScorer)
    assert score > 0.63


def test_scoring_func_fails_if_invalid_scorer_is_given():
    with pytest.raises(MLToolingError):
        get_scoring_func('invalid_scorer')
