# NOTE: Add more pathfinders here, if needed
from stara_maze_generator.pathfinder.bfs import BFS
from enum import Enum, auto


class Pathfinder(Enum):
    BFS = auto()


__all__ = ["Pathfinder", "BFS"]
