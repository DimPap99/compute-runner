"""Submit and observe Kaggle workloads without authentication side effects on import."""

from .models import BatchRecord, Config, JobRecord, JobSpec

__all__ = ["AgentClient", "BatchRecord", "Client", "Config", "JobRecord", "JobSpec"]


def __getattr__(name):
    if name == "AgentClient":
        from .agent import AgentClient

        return AgentClient
    if name == "Client":
        from .client import Client

        return Client
    raise AttributeError(name)
