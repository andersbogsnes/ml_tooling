from git import Repo, InvalidGitRepositoryError
import pathlib


class MLUtilsError(Exception):
    pass


def get_git_hash():
    try:
        repo = Repo(search_parent_directories=True)
    except InvalidGitRepositoryError:
        return ''
    return repo.head.object.hexsha


def find_model_file(path: str) -> pathlib.Path:
    """
    Helper to find a model file in a given directory.
    If path is a directory - returns newest model that matches the git hash
    :param path: dir or path to model
    :return:
    """
    path = pathlib.Path(path)

    if path.is_file():
        return path

    git_hash = get_git_hash()
    newest_match = max(path.glob(f'*_{git_hash}.pkl'), key=lambda x: x.stat().st_mtime)
    return newest_match


def get_model_name(clf) -> str:
    if clf.__class__.__name__ == 'Pipeline':
        return clf.steps[-1].__class__.__name__

    return clf.__class__.__name__
