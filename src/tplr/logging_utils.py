# Standard library
import logging
import time

# Third party
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler


def T() -> float:
    """Returns the current time in seconds since the epoch."""
    return time.time()


def P(window: int, duration: float) -> str:
    """Formats a log prefix with the window number and duration."""
    return f"[steel_blue]{window}[/steel_blue] ([grey63]{duration:.2f}s[/grey63])"


# Configure the root logger
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(
            markup=True,
            rich_tracebacks=True,
            highlighter=NullHighlighter(),
            show_level=False,
            show_time=True,
            show_path=False,
        )
    ],
)

logger = logging.getLogger("loco")
logger.setLevel(logging.INFO)
logger.propagate = True
logger.handlers.clear()
logger.addHandler(
    RichHandler(
        markup=True,
        rich_tracebacks=True,
        highlighter=NullHighlighter(),
        show_level=False,
        show_time=True,
        show_path=False,
    )
)


def debug() -> None:
    """Sets the logger level to DEBUG."""
    logger.setLevel(logging.DEBUG)


def trace() -> None:
    """Sets the logger level to TRACE."""
    TRACE_LEVEL_NUM = 5
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

    def trace_method(self, message, *args, **kws) -> None:
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kws)

    logging.Logger.trace = trace_method
    logger.setLevel(TRACE_LEVEL_NUM)


__all__ = [
    "logger",
    "debug",
    "trace",
    "P",
    "T",
]
