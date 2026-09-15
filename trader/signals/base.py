from abc import ABC, abstractmethod

import pandas as pd


class Signal(ABC):
    """A layer-2 signal: events in, one conviction score per event out.

    Scores are in [0, 1] (0.5 = no edge either way) and NaN where the
    signal has no opinion. A score may only use information available at
    the event's entry date — the validation gate assumes it.
    """

    name: str

    @abstractmethod
    def score(self, events: pd.DataFrame) -> pd.Series:
        ...
