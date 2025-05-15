# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
from typing import Any, Dict, List, Optional, Union

from camel.configs import ANTHROPIC_API_PARAMS, AnthropicConfig
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import ChatCompletion, ModelType
from camel.utils import (
    AnthropicTokenCounter,
    BaseTokenCounter,
    api_keys_required,
    dependencies_required,
)


class AnthropicModel(BaseModelBackend):
    r"""Anthropic API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, one of CLAUDE_* series.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into Anthropic.messages.create().  If
            :obj:`None`, :obj:`AnthropicConfig().as_dict()` will be used.
            (default::obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the Anthropic service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the Anthropic service.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`AnthropicTokenCounter`
            will be used. (default: :obj:`None`)
    """

    @dependencies_required('anthropic')
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        from anthropic import Anthropic

        if model_config_dict is None:
            model_config_dict = AnthropicConfig().as_dict()
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        url = url or os.environ.get("ANTHROPIC_API_BASE_URL")
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self.client = Anthropic(api_key=self._api_key, base_url=self._url)

    def _convert_response_from_anthropic_to_openai(self, response):
        # openai ^1.0.0 format, reference openai/types/chat/chat_completion.py
        obj = ChatCompletion.construct(
            id=None,
            choices=[
                dict(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": response.content[0].text,
                    },
                    finish_reason=response.stop_reason,
                )
            ],
            created=None,
            model=response.model,
            object="chat.completion",
        )
        return obj

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = AnthropicTokenCounter()
        return self._token_counter

    def count_tokens_from_prompt(self, prompt: str) -> int:
        r"""Count the number of tokens from a prompt.

        Args:
            prompt (str): The prompt string.

        Returns:
            int: The number of tokens in the prompt.
        """
        return self.client.count_tokens(prompt)

    @api_keys_required("ANTHROPIC_API_KEY")
    def run(
        self,
        messages: List[OpenAIMessage],
    ):
        r"""Run inference of Anthropic chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            ChatCompletion: Response in the OpenAI API format.
        """
        from anthropic import NOT_GIVEN

        if messages[0]["role"] == "system":
            sys_msg = str(messages.pop(0)["content"])
        else:
            sys_msg = NOT_GIVEN  # type: ignore[assignment]
        response = self.client.messages.create(
            model=self.model_type,
            system=sys_msg,
            messages=messages,  # type: ignore[arg-type]
            **self.model_config_dict,
        )

        # format response to openai format
        response = self._convert_response_from_anthropic_to_openai(response)

        return response

    def check_model_config(self):
        r"""Check whether the model configuration is valid for anthropic
        model backends.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to OpenAI API, or it does not contain
                :obj:`model_path` or :obj:`server_url`.
        """
        for param in self.model_config_dict:
            if param not in ANTHROPIC_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Anthropic model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get("stream", False)
