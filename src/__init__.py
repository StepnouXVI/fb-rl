"""Package initialization for src."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline_repo"))

try:
    from src.contexts import (
        BaseMazeContext,
        TopologicalPathContext,
        SequenceAttentionContext,
    )
    from src.stages import (
        PipelineStage,
        DijkstraPathBuilder,
        PathDatabaseLogger,
        StepDatabaseLogger,
        AttentionFilter,
        SubgoalSelector,
        SequenceAttentionTranslator,
        SingleWaypointTranslator,
        DirectIntentionTranslator,
        HighLevelActor,
        LowLevelActor,
    )
    from src.agent import (
        StagedAgent,
        create_baseline_agent,
        create_dijkstra_teacher_agent,
        create_single_waypoint_agent,
        create_sequence_attention_agent,
        create_direct_intention_agent,
    )
except ImportError:
    pass
