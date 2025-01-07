import pytest
import numpy as np
from stara_maze_generator.vmaze import VMaze
from stara_maze_generator.pathfinder.bfs import BFS


class TestBFS:
    @pytest.fixture
    def simple_maze(self):
        """Create a simple 4x4 maze for testing."""
        maze = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3), min_valid_paths=1)
        # Set up a simple maze layout (1 = passage, 0 = wall):
        # S 1 1 1
        # 0 0 1 1
        # 1 1 1 0
        # 1 0 1 G
        maze.maze_map = np.array(
            [[1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1]]
        )
        return maze

    @pytest.fixture
    def bfs_pathfinder(self, simple_maze):
        """Create a BFS pathfinder instance with the simple maze."""
        return BFS(simple_maze)

    def test_find_path_exists(self, simple_maze, bfs_pathfinder):
        """Test that BFS finds a valid path when one exists."""
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        assert path is not None
        assert len(path) > 0
        assert path[0] == tuple(simple_maze.start)
        assert path[-1] == tuple(simple_maze.goal)
        # Verify each step in path is valid (connected and passable)
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            # Check that steps are adjacent
            assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
            # Check that both positions are passable
            assert simple_maze.maze_map[curr] == 1
            assert simple_maze.maze_map[next_pos] == 1

    def test_find_path_no_path(self, simple_maze, bfs_pathfinder):
        """Test that BFS returns None when no path exists."""
        # Block all possible paths to goal
        simple_maze.maze_map[0, 1:] = 0  # Block top row
        simple_maze.maze_map[1:, -1] = 0  # Block right column
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        assert path is None

    def test_find_path_shortest(self, simple_maze, bfs_pathfinder):
        """Test that BFS finds a shortest path."""
        # Create a maze with a clear shortest path length
        simple_maze.maze_map = np.array(
            [
                [1, 1, 0, 0],  # Only one shortest path possible
                [0, 1, 0, 0],  # through (0,0)->(0,1)->(1,1)->(1,2)->(1,3)->(2,3)->(3,3)
                [0, 1, 1, 1],
                [0, 0, 0, 1],
            ]
        )
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        assert path is not None
        # The shortest path must be exactly 7 steps
        assert len(path) == 7
        # Verify it's a valid path
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            # Check that steps are adjacent
            assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
            # Check that both positions are passable
            assert simple_maze.maze_map[curr] == 1
            assert simple_maze.maze_map[next_pos] == 1

    def test_path_updates_maze_path(self, simple_maze, bfs_pathfinder):
        """Test that finding a path updates the maze's path attribute."""
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        assert path is not None
        assert simple_maze.path == path

    def test_visited_cells_not_revisited(self, simple_maze, bfs_pathfinder):
        """Test that BFS doesn't revisit already visited cells."""
        # Create a maze with a loop
        simple_maze.maze_map = np.array(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        # Count occurrences of each position in path
        position_counts = {}
        for pos in path:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        # Each position should only appear once
        assert all(count == 1 for count in position_counts.values())
