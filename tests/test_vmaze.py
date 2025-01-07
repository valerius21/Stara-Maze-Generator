import pytest
import numpy as np
from stara_maze_generator.vmaze import VMaze
from stara_maze_generator.pathfinder import Pathfinder


class TestVMaze:
    @pytest.fixture
    def basic_maze(self):
        """Create a basic 4x4 maze for testing."""
        maze = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))
        maze.maze_map = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]]
        )
        return maze

    def test_initialization(self):
        """Test that VMaze initializes with correct parameters."""
        maze = VMaze(seed=42, size=10, start=(0, 0), goal=(9, 9))
        assert maze.rows == 10
        assert maze.cols == 10
        assert np.array_equal(maze.start, np.array([0, 0]))
        assert np.array_equal(maze.goal, np.array([9, 9]))
        assert maze.seed == 42
        assert maze.pathfinding_algorithm == Pathfinder.BFS

    def test_initialization_validation(self):
        """Test that initialization validates parameters."""
        with pytest.raises(ValueError, match="size must be at least 4"):
            VMaze(seed=42, size=3, start=(0, 0), goal=(2, 2))

        with pytest.raises(IndexError):
            maze = VMaze(seed=42, size=4, start=(0, 0), goal=(4, 4))
            maze.maze_map[4, 4] = 1  # This will raise IndexError

    def test_maze_connectivity(self, basic_maze):
        """Test that maze has a valid path from start to goal."""
        path = basic_maze.find_path()
        assert path is not None, "Maze should have a valid path"
        assert tuple(basic_maze.start) == path[0]
        assert tuple(basic_maze.goal) == path[-1]

        # Verify path is valid
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
            assert basic_maze.maze_map[curr] == 1
            assert basic_maze.maze_map[next_pos] == 1

    def test_multiple_paths(self):
        """Test that maze can have multiple valid paths."""
        maze = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))
        # Create a maze with multiple possible paths
        maze.maze_map = np.array(
            [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )

        # Find first path
        original_path = maze.find_path()
        assert original_path is not None

        # Block the first path and verify another exists
        for x, y in original_path[1:-1]:
            maze.maze_map[x, y] = 0

        # Should still find a different path
        new_path = maze.find_path()
        assert new_path is not None
        assert new_path != original_path

    def test_get_cell_neighbours_center(self, basic_maze):
        """Test getting neighbors for a center cell."""
        neighbors = basic_maze.get_cell_neighbours(1, 1)
        assert len(neighbors) == 4
        up, down, left, right = neighbors
        assert up == (0, 1, 1)
        assert down == (2, 1, 1)
        assert left == (1, 0, 0)
        assert right == (1, 2, 0)

    def test_get_cell_neighbours_corner(self, basic_maze):
        """Test getting neighbors for corner cells."""
        neighbors = basic_maze.get_cell_neighbours(0, 0)
        assert len(neighbors) == 4
        assert neighbors[0] is None
        assert neighbors[2] is None
        assert neighbors[1] == (1, 0, 0)
        assert neighbors[3] == (0, 1, 1)

        neighbors = basic_maze.get_cell_neighbours(3, 3)
        assert len(neighbors) == 4
        assert neighbors[1] is None
        assert neighbors[3] is None
        assert neighbors[0] == (2, 3, 1)
        assert neighbors[2] == (3, 2, 0)

    def test_export_html(self, basic_maze, tmp_path):
        """Test HTML export functionality."""
        export_path = tmp_path / "maze.html"
        basic_maze.export_html(export_path)

        assert export_path.exists()
        content = export_path.read_text()
        assert "<html>" in content
        assert "<body>" in content
        assert f"Maze #{basic_maze.seed}" in content
        assert "cell-start" in content
        assert "cell-goal" in content

    def test_export_html_with_solution(self, basic_maze, tmp_path):
        """Test HTML export with solution path."""
        path = basic_maze.find_path()
        assert path is not None
        export_path = tmp_path / "maze_solution.html"
        basic_maze.export_html(export_path, draw_solution=True)

        assert export_path.exists()
        content = export_path.read_text()
        assert "cell-path" in content

    def test_repr(self, basic_maze):
        """Test string representation of maze."""
        expected = "VMaze(rows=4, cols=4, start=[0 0], goal=[3 3])"
        assert str(basic_maze) == expected
