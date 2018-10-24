import pytest

from ml_tooling.utils import get_git_hash, find_model_file, _is_percent


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
