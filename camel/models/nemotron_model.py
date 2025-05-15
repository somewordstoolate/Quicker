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
from typing import List, Optional, Union

from openai import OpenAI

from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import ChatCompletion, ModelType
from camel.utils import (
    BaseTokenCounter,
    api_keys_required,
)


class NemotronModel(BaseModelBackend):
    r"""Nemotron model API backend with OpenAI compatibility.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        api_key (Optional[str], optional): The API key for authenticating with
            the Nvidia service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the Nvidia service.
            (default: :obj:`https://integrate.api.nvidia.com/v1`)

    Notes:
        Nemotron model doesn't support additional model config like OpenAI.
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        url = url or os.environ.get(
            "NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        super().__init__(model_type, {}, api_key, url)
        self._client = OpenAI(
            timeout=60,
            max_retries=3,
            base_url=self._url,
            api_key=self._api_key,
        )

    @api_keys_required("NVIDIA_API_KEY")
    def run(
        self,
        messages: List[OpenAIMessage],
    ) -> ChatCompletion:
        r"""Runs inference of OpenAI chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list.

        Returns:
            ChatCompletion.
        """
        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
        )
        return response

    @property
    def token_counter(self) -> BaseTokenCounter:
        raise NotImplementedError(
            "Nemotron model doesn't support token counter."
        )

    def check_model_config(self):
        raise NotImplementedError(
            "Nemotron model doesn't support model config."
        )
