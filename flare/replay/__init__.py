"""Log replay — stream real log files through the detection pipeline at controlled speed."""

from flare.replay.replayer import LogReplayer, WindowResult
from flare.replay.shuffler import shuffled_stream

__all__ = ["LogReplayer", "WindowResult", "shuffled_stream"]
