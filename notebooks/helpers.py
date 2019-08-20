import numpy as np
import re

title_map = {
    "Mr.": "Mr",
    "Miss.": "Miss",
    "Mrs.": "Mrs",
    "Master.": "Master",
    "Dr.": "Other",
    "Rev.": "Other",
    "Major.": "Other",
    "Col.": "Other",
    "Mlle.": "Miss",
    "Lady.": "Other",
    "Jonkheer.": "Other",
    "Ms.": "Miss",
    "Capt.": "Other",
    "Don.": "Other",
    "Sir.": "Other",
    "Countess": "Other",
    "Mme.": "Mrs",
}

match = re.compile(r"([A-Z][a-z]*\.)")


def binarize_na(series):
    return np.where(series.isna(), 1, 0)


def convert_to_title(series):
    return series.str.extract(match, expand=False).map(title_map)


def create_ticket_letters(series):
    return (
        series.str.extract(r"(.*\s)", expand=False)
        .str.replace(".", "")
        .str.replace(" ", "")
        .str.strip()
        .fillna("None")
    )


def consolidate_ticket_letters(series):
    return series.where(series.isin(["FCC", "PC"]), "Other")


def is_missing(series):
    return series.isna() * 1


def get_cabin_letter(value):
    if isinstance(value, list):
        return value[0][0]
    return "X"


def extract_cabin_letter(series):
    return series.str.split(" ").apply(get_cabin_letter)


def get_num_cabins(value):
    if isinstance(value, list):
        return len(value)
    return 0


def extract_num_cabins(series):
    return series.str.split(" ").apply(get_num_cabins)
