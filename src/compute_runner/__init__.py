"""Submit and observe compute workloads without authentication side effects on import."""

from .models import Account, BatchRecord, Config, JobRecord, JobSpec

__all__ = ["Account", "AgentClient", "BatchRecord", "Client", "Config", "JobRecord", "JobSpec"]


def __getattr__(name):
    if name == "AgentClient":
        from .agent import AgentClient

        return AgentClient
    if name == "Client":
        from .client import Client

        return Client
    raise AttributeError(name)
