import pytest
import numpy as np
from stara_maze_generator.vmaze import VMaze
from stara_maze_generator.pathfinder.base import PathfinderBase


class TestPathfinderBase:
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
    def pathfinder(self, simple_maze):
        """Create a PathfinderBase instance with the simple maze."""

        class ConcretePathfinder(PathfinderBase):
            def find_path(self, start, goal):
                return [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 2), (3, 2), (3, 3)]

        return ConcretePathfinder(simple_maze)

    def test_initialization(self, simple_maze):
        """Test that PathfinderBase initializes correctly."""
        pathfinder = PathfinderBase(simple_maze)
        assert pathfinder.maze == simple_maze

    def test_find_path_not_implemented(self, simple_maze):
        """Test that base find_path raises NotImplementedError."""
        pathfinder = PathfinderBase(simple_maze)
        with pytest.raises(NotImplementedError):
            pathfinder.find_path((0, 0), (3, 3))

    def test_get_cell_neighbours(self, simple_maze, pathfinder):
        """Test that get_cell_neighbours returns correct neighbors."""
        # Test center cell (2,2)
        neighbors = pathfinder.maze.get_cell_neighbours(2, 2)
        assert len(neighbors) == 4  # Should have 4 neighbors
        # Check up, down, left, right neighbors exist and have correct values
        assert neighbors[0] == (1, 2, 1)  # up
        assert neighbors[1] == (3, 2, 1)  # down
        assert neighbors[2] == (2, 1, 1)  # left
        assert neighbors[3] == (2, 3, 0)  # right

    def test_edge_cell_neighbours(self, simple_maze, pathfinder):
        """Test get_cell_neighbours for edge cells."""
        # Test top-left corner (0,0)
        neighbors = pathfinder.maze.get_cell_neighbours(0, 0)
        assert len(neighbors) == 4
        assert neighbors[0] is None  # up (out of bounds)
        assert neighbors[1] == (1, 0, 0)  # down
        assert neighbors[2] is None  # left (out of bounds)
        assert neighbors[3] == (0, 1, 1)  # right

    def test_boundary_validation(self, simple_maze):
        """Test that maze size validation works."""
        with pytest.raises(ValueError, match="size must be at least 4"):
            VMaze(seed=42, size=3, start=(0, 0), goal=(2, 2))
