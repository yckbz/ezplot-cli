class EzPlotError(Exception):
    """Base exception for ezplot failures."""


class DataFormatError(EzPlotError):
    """Raised when the input file shape or values are invalid."""


class DataSelectionError(EzPlotError):
    """Raised when a requested column cannot be resolved."""
