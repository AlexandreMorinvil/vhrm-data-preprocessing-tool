from PyQt6.QtWidgets import QSplitter, QWidget


LEFT_PANE_MIN_WIDTH = 380
RIGHT_PANE_MIN_WIDTH = 480


def configure_main_splitter(
    splitter: QSplitter,
    left: QWidget,
    right: QWidget,
    left_stretch: int = 1,
    right_stretch: int = 3,
) -> None:
    left.setMinimumWidth(LEFT_PANE_MIN_WIDTH)
    right.setMinimumWidth(RIGHT_PANE_MIN_WIDTH)
    splitter.setChildrenCollapsible(False)
    splitter.setStretchFactor(0, left_stretch)
    splitter.setStretchFactor(1, right_stretch)
    splitter.setSizes([LEFT_PANE_MIN_WIDTH, RIGHT_PANE_MIN_WIDTH * 2])