import logging
from typing import Dict, List, Optional

from camel.societies.workforce.base import BaseNode
from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from utils.Search_Strategy.worker import SingleAgentWorkerWithLog


# class WorkforceWithLog(Workforce):
#     r"""A system where multiple workder nodes (agents) cooperate together
#     to solve tasks. It can assign tasks to workder nodes and also take
#     strategies such as create new worker, decompose tasks, etc. to handle
#     situations when the task fails.

#     Args:
#         description (str): Description of the node.
#         children (Optional[List[BaseNode]], optional): List of child nodes
#             under this node. Each child node can be a worker node or
#             another workforce node. (default: :obj:`None`)
#         coordinator_agent_kwargs (Optional[Dict], optional): Keyword
#             arguments for the coordinator agent, e.g. `model`, `api_key`,
#             `tools`, etc. (default: :obj:`None`)
#         task_agent_kwargs (Optional[Dict], optional): Keyword arguments for
#             the task agent, e.g. `model`, `api_key`, `tools`, etc.
#             (default: :obj:`None`)
#         new_worker_agent_kwargs (Optional[Dict]): Default keyword arguments
#             for the worker agent that will be created during runtime to
#             handle failed tasks, e.g. `model`, `api_key`, `tools`, etc.
#             (default: :obj:`None`)
#     """

#     def __init__(
#         self,
#         description: str,
#         children: Optional[List[BaseNode]] = None,
#         coordinator_agent_kwargs: Optional[Dict] = None,
#         task_agent_kwargs: Optional[Dict] = None,
#         new_worker_agent_kwargs: Optional[Dict] = None,
#     ) -> None:
#         super().__init__(description)


def add_single_agent_worker_with_log(
    self, description: str, worker: ChatAgent, logger: logging.Logger
) -> Workforce:
    r"""Add a worker node to the workforce that uses a single agent.

    Args:
        description (str): Description of the worker node.
        worker (ChatAgent): The agent to be added.

    Returns:
        Workforce: The workforce node itself.
    """
    worker_node = SingleAgentWorkerWithLog(description, worker, logger)
    self._children.append(worker_node)
    return self
