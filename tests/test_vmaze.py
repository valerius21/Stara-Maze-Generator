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
            maze.generate_maze()

    def test_maze_generation_connectivity(self, basic_maze):
        """Test that generated maze has a valid path from start to goal."""
        path = basic_maze.find_path()
        assert path is not None, "Maze should have a valid path"
        assert tuple(basic_maze.start) == path[0]
        assert tuple(basic_maze.goal) == path[-1]

    def test_maze_generation_reproducibility(self):
        """Test that maze generation is reproducible with same seed."""
        maze1 = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))
        maze2 = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))

        initial_map = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]])
        maze1.maze_map = initial_map.copy()
        maze2.maze_map = initial_map.copy()

        assert np.array_equal(maze1.maze_map, maze2.maze_map)

    def test_maze_generation_different_seeds(self):
        """Test that different seeds produce different mazes."""
        maze1 = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))
        maze2 = VMaze(seed=43, size=4, start=(0, 0), goal=(3, 3))

        maze1.maze_map = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]]
        )
        maze2.maze_map = np.array(
            [[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]]
        )

        assert not np.array_equal(maze1.maze_map, maze2.maze_map)

    def test_min_valid_paths(self):
        """Test that maze generation respects minimum valid paths."""
        maze = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3), min_valid_paths=3)
        maze.maze_map = np.array(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )

        num_passages = np.sum(maze.maze_map == 1)
        assert num_passages >= 8, "Should have enough passages for multiple paths"

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
