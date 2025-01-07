from typing import Protocol


class VMazeProtocol(Protocol):
    """Protocol for VMaze to avoid circular imports."""

    rows: int
    cols: int
    maze_map: list
    path: list

    def get_cell_neighbours(self, x: int, y: int) -> tuple[tuple[int, int, int]]: ...
