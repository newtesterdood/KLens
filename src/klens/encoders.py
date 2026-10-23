from __future__ import annotations

from typing import Dict, List, Protocol, Sequence, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextEncoder(Protocol):
    def encode_grid(
        self,
        text_grid: Dict[Tuple[int, str], str],
        rounds: Sequence[int],
        speakers: Sequence[str],
    ) -> Tuple[np.ndarray, List[str]]:
        ...


class TfidfTextEncoder:
    def __init__(self, max_features: int = 30, stop_words: str = "english"):
        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer: TfidfVectorizer | None = None

    def encode_grid(
        self,
        text_grid: Dict[Tuple[int, str], str],
        rounds: Sequence[int],
        speakers: Sequence[str],
    ) -> Tuple[np.ndarray, List[str]]:
        T = len(rounds)
        N = len(speakers)

        all_texts = [text_grid.get((r, s), "").strip() for r in rounds for s in speakers]
        non_empty = [t for t in all_texts if t]
        if not non_empty:
            return np.zeros((T, N, 1)), ["none"]

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
        )
        self.vectorizer.fit(non_empty)
        feature_names = list(self.vectorizer.get_feature_names_out())
        d = len(feature_names)
        if d == 0:
            return np.zeros((T, N, 1)), ["empty"]

        Z = np.zeros((T, N, d))
        for t_idx, r in enumerate(rounds):
            for s_idx, s in enumerate(speakers):
                txt = text_grid.get((r, s), "").strip()
                if txt:
                    Z[t_idx, s_idx, :] = self.vectorizer.transform([txt]).toarray()

        return Z, feature_names
