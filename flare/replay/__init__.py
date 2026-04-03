"""Log replay — stream real log files through the detection pipeline at controlled speed."""

from flare.replay.replayer import LogReplayer, WindowResult

__all__ = ["LogReplayer", "WindowResult"]
