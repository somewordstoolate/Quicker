from camel.societies.workforce.single_agent_worker import SingleAgentWorker
import ast
from typing import Any, List
import logging

from colorama import Fore
from camel.agents import ChatAgent
from camel.messages.base import BaseMessage
from camel.societies.workforce.prompts import PROCESS_TASK_PROMPT
from camel.societies.workforce.utils import TaskResult
from camel.tasks.task import Task, TaskState
from camel.utils import print_text_animated


class SingleAgentWorkerWithLog(SingleAgentWorker):
    def __init__(
        self,
        description: str,
        worker: ChatAgent,
        logger: logging.Logger,
    ) -> None:
        super().__init__(description, worker)
        self.logger = logger

    async def _process_task(self, task: Task, dependencies: List[Task]) -> TaskState:
        r"""Processes a task with its dependencies.

        This method asynchronously processes a given task, considering its
        dependencies, by sending a generated prompt to a worker. It updates
        the task's result based on the agent's response.

        Args:
            task (Task): The task to process, which includes necessary details
                like content and type.
            dependencies (List[Task]): Tasks that the given task depends on.

        Returns:
            TaskState: `TaskState.DONE` if processed successfully, otherwise
                `TaskState.FAILED`.
        """
        dependency_tasks_info = self._get_dep_tasks_info(dependencies)
        prompt = PROCESS_TASK_PROMPT.format(
            content=task.content,
            dependency_tasks_info=dependency_tasks_info,
            additional_info=task.additional_info,
        )
        req = BaseMessage.make_user_message(
            role_name="User",
            content=prompt,
        )
        try:
            response = self.worker.step(req, response_format=TaskResult)
        except Exception as e:
            print(
                f"{Fore.RED}Error occurred while processing task {task.id}:"
                f"\n{e}{Fore.RESET}"
            )
            self.logger.error(f"Error occurred while processing task {task.id}:\n{e}")
            return TaskState.FAILED

        print(f"======\n{Fore.GREEN}Reply from {self}:{Fore.RESET}")
        self.logger.info(f"======\nReply from {self}:")
        self.logger.info(f'dependency_tasks_info:\n{dependency_tasks_info}\n')
        self.logger.info(f'task.content:\n{task.content}\n')

        result_dict = ast.literal_eval(response.msg.content)
        task_result = TaskResult(**result_dict)

        color = Fore.RED if task_result.failed else Fore.GREEN
        print_text_animated(
            f"\n{color}{task_result.content}{Fore.RESET}\n======",
            delay=0.005,
        )
        self.logger.info(task_result.content + "\n======")

        if task_result.failed:
            return TaskState.FAILED

        task.result = task_result.content
        return TaskState.DONE
