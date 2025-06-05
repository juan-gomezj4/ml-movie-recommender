from typing import Any, ClassVar

from numpy import nan, ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MultiLabelBinarizer,
    OneHotEncoder,
)


class MovieFeaturePreprocessor:
    NUM_COLS: ClassVar[list[str]] = [
        "popularity",
        "vote_average",
        "vote_count",
        "runtime",
        "budget",
        "revenue",
        "is_popular",
    ]
    CAT_COLS: ClassVar[list[str]] = ["original_language"]
    MULTI_LABEL_CAT_COLS: ClassVar[list[str]] = ["genres", "spoken_languages"]

    # ruff: noqa: RUF001
    ORIGINAL_LANGUAGE_MAPPINGS: ClassVar[dict[str, str]] = {
        "fr": "European (Romance)",
        "es": "European (Romance)",
        "en": "European (Germanic)",
        "te": "South Asian",
        "de": "European (Germanic)",
        "hi": "South Asian",
        "ja": "East Asian",
        "nl": "European (Germanic)",
        "th": "Southeast Asian",
        "id": "Southeast Asian",
        "ht": "European (Romance)",
        "it": "European (Romance)",
        "ta": "South Asian",
        "ml": "South Asian",
        "fi": "European (Other)",
        "ko": "East Asian",
        "bg": "European (Slavic)",
        "ca": "European (Romance)",
        "pt": "European (Romance)",
        "tr": "Middle Eastern/Central Asian",
        "no": "European (Germanic)",
        "tl": "Southeast Asian",
        "da": "European (Germanic)",
        "zu": "African",
        "sv": "European (Germanic)",
        "pl": "European (Slavic)",
        "uk": "European (Slavic)",
        "zh": "East Asian",
        "ru": "European (Slavic)",
        "kn": "South Asian",
        "xx": "Unknown/Other",
        "cn": "East Asian",
        "ar": "Middle Eastern/Central Asian",
        "hu": "European (Other)",
        "fa": "Middle Eastern/Central Asian",
        "mn": "East Asian",
        "yo": "African",
        "ro": "European (Romance)",
        "sk": "European (Slavic)",
        "jv": "Southeast Asian",
        "cs": "European (Slavic)",
        "ur": "South Asian",
        "pa": "South Asian",
        "is": "European (Germanic)",
        "hr": "European (Slavic)",
        "vi": "Southeast Asian",
        "lv": "European (Other)",
        "km": "Southeast Asian",
        "ms": "Southeast Asian",
        "kk": "Middle Eastern/Central Asian",
        "ka": "European (Other)",
        "ga": "European (Other)",
        "xh": "African",
        "ig": "African",
        "el": "European (Other)",
        "bn": "South Asian",
        "tt": "Middle Eastern/Central Asian",
        "gl": "European (Romance)",
        "mk": "European (Slavic)",
        "bo": "East Asian",
        "dz": "South Asian",
        "he": "Middle Eastern/Central Asian",
        "sr": "European (Slavic)",
        "ff": "African",
        "gu": "South Asian",
        "ab": "European (Other)",
        "et": "European (Other)",
        "kl": "European (Other)",
        "lt": "European (Other)",
        "se": "European (Other)",
        "eu": "European (Other)",
        "bs": "European (Slavic)",
        "lb": "European (Germanic)",
        "mi": "Southeast Asian",
        "hy": "Middle Eastern/Central Asian",
        "su": "Southeast Asian",
        "mt": "Middle Eastern/Central Asian",
        "sl": "European (Slavic)",
    }

    # ruff: noqa: RUF001
    SPOKEN_LANGUAGES_MAPPINGS: ClassVar[dict[str | float, str]] = {
        "Français": "European (Romance)",
        "Español": "European (Romance)",
        "English": "European (Germanic)",
        "Deutsch": "European (Germanic)",
        "हिन्दी": "South Asian",
        "广州话 / 廣州話": "East Asian",
        "日本語": "East Asian",
        "Italiano": "European (Romance)",
        "Pусский": "European (Slavic)",
        "Nederlands": "European (Germanic)",
        "isiZulu": "African",
        "ภาษาไทย": "Southeast Asian",
        "普通话": "East Asian",
        "Bahasa indonesia": "Southeast Asian",
        "": "Unknown/Other",
        "தமிழ்": "South Asian",
        "suomi": "European (Other)",
        "한국어/조선말": "East Asian",
        "български език": "European (Slavic)",
        "Català": "European (Romance)",
        "Türkçe": "Middle Eastern/Central Asian",
        "Português": "European (Romance)",
        "Norsk": "European (Germanic)",
        "Dansk": "European (Germanic)",
        "svenska": "European (Germanic)",
        "Lietuvių": "European (Other)",
        "Polski": "European (Slavic)",
        "తెలుగు": "South Asian",
        "עִבְרִית": "Middle Eastern/Central Asian",
        "Український": "European (Slavic)",
        "Latin": "European (Other)",  # Could also be considered Historical
        "?????": "Unknown/Other",
        "No Language": "Unknown/Other",
        "اردو": "South Asian",
        "العربية": "Middle Eastern/Central Asian",
        "Română": "European (Romance)",
        "Íslenska": "European (Germanic)",
        "Magyar": "European (Other)",
        "فارسی": "Middle Eastern/Central Asian",
        "Bahasa melayu": "Southeast Asian",
        "Galego": "European (Romance)",
        "ქართული": "European (Other)",  # Kartvelian is a unique family, grouped here for simplicity
        "euskera": "European (Other)",  # Language Isolate, grouped here
        "Èdè Yorùbá": "African",
        "Wolof": "African",
        "Gaeilge": "European (Other)",  # Celtic, grouped here
        "Hrvatski": "European (Slavic)",
        "ελληνικά": "European (Other)",  # Hellenic, grouped here
        "Slovenčina": "European (Slavic)",
        "πੰਜਾਬੀ": "South Asian",
        "Český": "European (Slavic)",
        "Tiếng Việt": "Southeast Asian",
        "Fulfulde": "African",
        "қазақ": "Middle Eastern/Central Asian",
        "Esperanto": "Unknown/Other",  # Constructed language
        "Èʋegbe": "African",
        "বাংলা": "South Asian",
        "پښتو": "Middle Eastern/Central Asian",
        "shqip": "European (Other)",  # Albanian, grouped here
        "Srpski": "European (Slavic)",
        "Afrikaans": "European (Germanic)",
        "Kiswahili": "African",
        "Eesti": "European (Other)",  # Uralic, grouped here
        "Slovenščina": "European (Slavic)",
        "Bamanankan": "African",
        "Azərbaycan": "Middle Eastern/Central Asian",
        "Bosanski": "European (Slavic)",
        "සිංහල": "South Asian",
        "Latviešu": "European (Other)",  # Baltic, grouped here
        "Malti": "Middle Eastern/Central Asian",
        nan: "Unknown/Other",
    }

    @classmethod
    def get_features_names(cls, _: Any, feature_names: ndarray) -> ndarray:  # TODO: Correct Any
        return feature_names

    @classmethod
    def map_lang(
        cls,
        X: DataFrame,
        col: str,
        mappings: dict[str, str],
    ) -> DataFrame:
        """
        Map language codes to broader categories.
        """
        X = X.copy()
        if X[col].dtype.name == "object":  # if it's a list
            X[col] = X[col].apply(lambda x: [mappings.get(item, "Unknown/Other") for item in x])
            return X
        X[col] = X[col].map(mappings)
        return X

    @classmethod
    def get_preprocessor(cls) -> ColumnTransformer:
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

        cat_pipe = Pipeline(
            steps=[
                (
                    "language mapper",
                    FunctionTransformer(
                        cls.map_lang,
                        kw_args={
                            "col": "original_language",
                            "mappings": cls.ORIGINAL_LANGUAGE_MAPPINGS,
                        },
                        feature_names_out=cls.get_features_names,
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one-hot", OneHotEncoder(drop="first")),
            ]
        )

        multi_label_genres_pipe = Pipeline(steps=[("binarizer", MultiLabelBinarizerTransformer())])

        multi_label_spoken_languages_pipe = Pipeline(
            steps=[
                (
                    "language mapper",
                    FunctionTransformer(
                        cls.map_lang,
                        kw_args={
                            "col": "spoken_languages",
                            "mappings": cls.SPOKEN_LANGUAGES_MAPPINGS,
                        },
                        feature_names_out=cls.get_features_names,
                    ),
                ),
                ("binarizer", MultiLabelBinarizerTransformer()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, cls.NUM_COLS),
                ("cat", cat_pipe, cls.CAT_COLS),
                ("genres", multi_label_genres_pipe, ["genres"]),
                (
                    "spoken_languages",
                    multi_label_spoken_languages_pipe,
                    ["spoken_languages"],
                ),
            ],
        )
        return preprocessor


class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """A custom transformer to apply MultiLabelBinarizer within a scikit-learn pipeline.

    This transformer is designed to be used with `ColumnTransformer` on a single
    column of a pandas DataFrame that contains lists of labels (multi-label data).
    It wraps the functionality of `sklearn.preprocessing.MultiLabelBinarizer` and
    provides a `get_feature_names_out` method compatible with scikit-learn pipelines.
    """

    def __init__(self) -> None:
        """Initializes the MultiLabelBinarizerTransformer."""
        self.mlb = MultiLabelBinarizer()

    def fit(self, X: DataFrame, y: Any = None) -> MultiLabelBinarizer:
        """Fits the MultiLabelBinarizer on the input data.

        Args:
            X: A pandas DataFrame slice with one column containing lists of labels.
            y: Ignored.

        Returns:
            self: Returns the instance itself.
        """
        self.mlb.fit(X.iloc[:, 0])
        return self

    def transform(self, X: DataFrame) -> Any:  # TODO: Fix Any
        """Transforms the input data using the fitted MultiLabelBinarizer.

        Args:
            X: A pandas DataFrame slice with one column containing lists of labels.

        Returns:
            numpy.ndarray: A sparse matrix representing the binarized labels.
        """
        # Transform the values of the column
        return self.mlb.transform(X.iloc[:, 0])

    def get_feature_names_out(self, _: Any = None) -> Any:  # TODO: Fix Any
        """Gets the output feature names after binarization.

        Args:
            input_features: Ignored.

        Returns:
            list: A list of strings representing the output feature names (the labels).
        """
        # Return the classes learned by the fitted MultiLabelBinarizer
        return self.mlb.classes_.tolist()
