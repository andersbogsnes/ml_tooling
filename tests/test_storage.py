import pytest

from ml_tooling.storage import Storage


def test_can_list_estimators():
    pass

def test_can_load_pickle_file():
    pass

def test_can_save_pickle_file():
    pass

def test_cannot_instantiate_an_abstract_baseclass():
    with pytest.raises(TypeError):
        Storage()