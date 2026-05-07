"""Cognitive task suite for COSYS-XNN validation."""

from tasks.cognitive_tasks import (
    n_back_task,
    stroop_task,
    iowa_gambling_task,
    morris_maze_task,
)

__all__ = ["n_back_task", "stroop_task", "iowa_gambling_task", "morris_maze_task"]
