"""Grounded multi-page DocQA agent: parser -> index -> planner -> orchestrator
-> evidence ledger/verifier -> official scorer, with a fully traced folder per
sample.  See ``usage/04_agent_qa.py`` and ``pipeline_v2.agent.runner``."""

from pipeline_v2.agent.common import AgentSettings as AgentSettings
from pipeline_v2.agent.runner import RunSpec as RunSpec
from pipeline_v2.agent.runner import run_sample as run_sample
