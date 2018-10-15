from ml_tools.baseclass.utils import get_git_hash, find_model_file


def test_get_git_hash_returns_correctly():
    hash = get_git_hash()
    assert isinstance(hash, str)
    assert 10 < len(hash)


def test_find_model_file_with_given_model_returns_correctly(tmpdir, monkeypatch):
    def mockreturn():
        return '1234'

    monkeypatch.setattr('ml_tools.baseclass.utils.get_git_hash', mockreturn)

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

    monkeypatch.setattr('ml_tools.baseclass.utils.get_git_hash', mockreturn)

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
