from typing import Callable, Dict, List, Literal, Optional, Union
import re
from camel.tasks import Task
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.prompts import TextPrompt
from utils.Search_Strategy.prompt import Search_Strategy_TASK_COMPOSE_PROMPT


def parse_search_strategy(msg_content: str) -> str:
    r"""Parse search strategy from a message.

    Args:
        msg_content (str): The content of the message.

    Returns:
        list: A list of search strategy.
    """
    pattern = r"<search strategy>(.*?)</search strategy>"
    search_strategy_content = re.findall(
        pattern, msg_content, re.DOTALL
    )  #! re.DOTALL: . matches any character, including a newline

    search_strategy = []
    for i, content in enumerate(search_strategy_content):
        search_strategy.append(content.strip())
    return search_strategy


class Search_Strategy_Task(Task):
    r"""A task for the search strategy workforce.

    Attributes:
        content (str): The content of the task.
        id (str): The unique identifier of the task.
        state (Literal['OPEN', 'RUNNING', 'DONE', 'DELETED']): The state of
            the task.
        type (str): The type of the task.
        parent (Optional[Task], optional): The parent task of the task.
            Defaults to None.
        subtasks (Optional[List[Task]], optional): The sub-tasks of the task.
            Defaults to None.
        result (Optional[str], optional): The result of the task. Defaults to
            None.
        additional_info (Optional[Dict], optional): Additional information
            about the task. Defaults to None.
    """

    search_strategy: Optional[list] = []

    def compose(
        self,
        agent: ChatAgent,
        template: TextPrompt = Search_Strategy_TASK_COMPOSE_PROMPT,
        result_parser: Optional[Callable[[str], str]] = parse_search_strategy,
    ):
        r"""compose task result by the sub-tasks.

        Args:
            agent (ChatAgent): An agent that used to compose the task result.
            template (TextPrompt, optional): The prompt template to compose
                task. If not provided, the default template will be used.
            result_parser (Callable[[str, str], List[Task]], optional): A
                function to extract Task from response.
        """

        if not self.subtasks:
            return

        sub_tasks_result = self.get_result()

        role_name = agent.role_name
        content = template.format(
            role_name=role_name,
            content=self.content,
            additional_info=self.additional_info,
            other_results=sub_tasks_result,
        )
        msg = BaseMessage.make_user_message(role_name=role_name, content=content)
        response = agent.step(msg)
        result = response.msg.content

        if result_parser:
            search_strategy = result_parser(result)
            self.update_search_strategy(search_strategy)
        self.update_result(result)

    def update_search_strategy(self, search_strategy: list):
        r'''Update the search strategy of the task.

        Args:
            search_strategy (list): The search strategy of the task.
        '''

        self.search_strategy = search_strategy
