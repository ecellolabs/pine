"""Grounded multi-page DocQA agent: parser -> index -> planner -> orchestrator
-> evidence ledger/verifier -> official scorer, with a fully traced folder per
sample.  See ``usage/02_agent_qa.py`` and ``pine.agent.runner``."""

from pine.agent.common import AgentSettings as AgentSettings
from pine.agent.runner import RunSpec as RunSpec
from pine.agent.runner import run_sample as run_sample
