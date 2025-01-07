#!/usr/bin/env python3

import argparse
from pathlib import Path
from time import time

import numpy as np
from loguru import logger

from stara_maze_generator.pathfinder import Pathfinder
from stara_maze_generator.vmaze import VMaze


def get_default_output(
    size: int, seed: int, min_valid_paths: int, pathfinding_algorithm: Pathfinder
) -> Path:
    """Generate default output filename based on maze settings."""
    return Path(
        f"{size}x{size}_seed{seed}_paths{min_valid_paths}_{pathfinding_algorithm.name}_maze.html"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and visualize mazes using Prim's algorithm"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=40,
        help="Size of the maze (creates a size x size grid)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible maze generation",
    )
    parser.add_argument(
        "--start",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("ROW", "COL"),
        help="Starting position coordinates (row, col)",
    )
    parser.add_argument(
        "--goal",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        help="Goal position coordinates (row, col). Defaults to (size-2, size-2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML file path. Defaults to size_seedX_pathsY_maze.html",
    )
    parser.add_argument(
        "--draw-solution",
        action="store_true",
        help="Draw the solution path in the output",
    )
    parser.add_argument(
        "--min-valid-paths",
        type=int,
        default=3,
        help="Minimum number of valid paths to ensure connectivity",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If goal not specified, use (size-2, size-2)
    if args.goal is None:
        args.goal = [args.size - 2, args.size - 2]

    # Set default output path if not specified
    if args.output is None:
        args.output = get_default_output(
            args.size, args.seed, args.min_valid_paths, Pathfinder.BFS
        )

    # Create and generate maze
    maze = VMaze(
        seed=args.seed,
        size=args.size,
        start=args.start,
        goal=args.goal,
        min_valid_paths=args.min_valid_paths,
    )

    start_time = time()
    maze.generate_maze(pathfinding_algorithm=Pathfinder.BFS)
    end_time = time()

    # Log generation info
    path = maze.find_path()
    logger.info(
        f"{Pathfinder.BFS}, len(path): {len(path) if path else 'No path found'}"
    )
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")

    # Export visualization
    maze.export_html(args.output, draw_solution=args.draw_solution)
    logger.info(f"Maze exported to {args.output}")


if __name__ == "__main__":
    main()
