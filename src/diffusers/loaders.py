# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import torch

from .models.attention_processor import LoRAAttnProcessor
from .utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    _get_model_file,
    deprecate,
    is_safetensors_available,
    is_transformers_available,
    logging,
)

from transformers import CLIPTokenizer
import copy
import random

if is_safetensors_available():
    import safetensors

if is_transformers_available():
    from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)


LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

TEXT_INVERSION_NAME = "learned_embeds.bin"
TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"


class AttnProcsLayers(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = key.split(".processor")[0] + ".processor"
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class UNet2DConditionLoadersMixin:
    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into `UNet2DConditionModel`. Attention processor layers have to be
        defined in
        [cross_attention.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
        and be a `torch.nn.Module` class.

        <Tip warning={true}>

            This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>
        """

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors"
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except IOError as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        attn_processors = {}

        is_lora = all("lora" in k for k in state_dict.keys())

        if is_lora:
            lora_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in lora_grouped_dict.items():
                rank = value_dict["to_k_lora.down.weight"].shape[0]
                cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
                hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

                attn_processors[key] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
                )
                attn_processors[key].load_state_dict(value_dict)

        else:
            raise ValueError(f"{model_file} does not seem to be in the correct format expected by LoRA training.")

        # set correct dtype & device
        attn_processors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in attn_processors.items()}

        # set layers
        self.set_attn_processor(attn_processors)

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = False,
        **kwargs,
    ):
        r"""
        Save an attention processor to a directory, so that it can be re-loaded using the
        `[`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]` method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        weight_name = weight_name or deprecate(
            "weights_name",
            "0.18.0",
            "`weights_name` is deprecated, please use `weight_name` instead.",
            take_from=kwargs,
        )
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = AttnProcsLayers(self.attn_processors)

        # Save the model
        state_dict = model_to_save.state_dict()

        if weight_name is None:
            if safe_serialization:
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = LORA_WEIGHT_NAME

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")


class TextualInversionLoaderMixin:
    r"""
    Mixin class for loading textual inversion tokens and embeddings to the tokenizer and text encoder.
    """

    def maybe_convert_prompt(self, prompt: Union[str, List[str]], tokenizer: "PreTrainedTokenizer"):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.

        Parameters:
            prompt (`str` or list of `str`):
                The prompt or prompts to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str` or list of `str`: The converted prompt
        """
        if not isinstance(prompt, List):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]

        if not isinstance(prompt, List):
            return prompts[0]

        return prompts

    def _maybe_convert_prompt(self, prompt: str, tokenizer: "PreTrainedTokenizer"):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.

        Parameters:
            prompt (`str`):
                The prompt to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str`: The converted prompt
        """
        tokens = tokenizer.tokenize(prompt)
        for token in tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f"{token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def load_textual_inversion(
        self, pretrained_model_name_or_path: Union[str, Dict[str, torch.Tensor]], token: Optional[str] = None, **kwargs
    ):
        r"""
        Load textual inversion embeddings into the text encoder of stable diffusion pipelines. Both `diffusers` and
        `Automatic1111` formats are supported.

        <Tip warning={true}>

            This function is experimental and might change in the future.

        </Tip>

        Parameters:
             pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like
                      `"sd-concepts-library/low-poly-hd-logos-icons"`.
                    - A path to a *directory* containing textual inversion weights, e.g.
                      `./my_text_inversion_directory/`.
            weight_name (`str`, *optional*):
                Name of a custom weight file. This should be used in two cases:

                    - The saved textual inversion file is in `diffusers` format, but was saved under a specific weight
                      name, such as `text_inv.bin`.
                    - The saved textual inversion file is in the "Automatic1111" form.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>
        """
        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PreTrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PreTrainedTokenizer` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        if not hasattr(self, "text_encoder") or not isinstance(self.text_encoder, PreTrainedModel):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` of type `PreTrainedModel` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors"
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
            allow_pickle = True

        user_agent = {
            "file_type": "text_inversion",
            "framework": "pytorch",
        }

        # 1. Load textual inversion file
        model_file = None
        # Let's first try to load .safetensors weights
        if (use_safetensors and weight_name is None) or (
            weight_name is not None and weight_name.endswith(".safetensors")
        ):
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=weight_name or TEXT_INVERSION_NAME_SAFE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = safetensors.torch.load_file(model_file, device="cpu")
            except Exception as e:
                if not allow_pickle:
                    raise e

                model_file = None

        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=weight_name or TEXT_INVERSION_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = torch.load(model_file, map_location="cpu")

        # 2. Load token and embedding correcly from file
        if isinstance(state_dict, torch.Tensor):
            if token is None:
                raise ValueError(
                    "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                )
            embedding = state_dict
        elif len(state_dict) == 1:
            # diffusers
            loaded_token, embedding = next(iter(state_dict.items()))
        elif "string_to_param" in state_dict:
            # A1111
            loaded_token = state_dict["name"]
            embedding = state_dict["string_to_param"]["*"]

        if token is not None and loaded_token != token:
            logger.warn(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
        else:
            token = loaded_token

        embedding = embedding.to(dtype=self.text_encoder.dtype, device=self.text_encoder.device)

        # 3. Make sure we don't mess up the tokenizer or text encoder
        vocab = self.tokenizer.get_vocab()
        if token in vocab:
            raise ValueError(
                f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
            )
        elif f"{token}_1" in vocab:
            multi_vector_tokens = [token]
            i = 1
            while f"{token}_{i}" in self.tokenizer.added_tokens_encoder:
                multi_vector_tokens.append(f"{token}_{i}")
                i += 1

            raise ValueError(
                f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
            )

        is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1

        if is_multi_vector:
            tokens = [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
            embeddings = [e for e in embedding]  # noqa: C416
        else:
            tokens = [token]
            embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

        # add tokens and get ids
        self.tokenizer.add_tokens(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # resize token embeddings and set new embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        for token_id, embedding in zip(token_ids, embeddings):
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding

        logger.info("Loaded textual inversion embedding for {token}.")


    def load_textual_inversion_embeddings(
        self,
        embedding_path_dict_or_list: Union[Dict[str, str], List[Dict[str, str]]],
        allow_replacement: bool = False,
    ):
        r"""
        Loads textual inversion embeddings and adds them to the tokenizer's vocabulary and the text encoder's embeddings.
        Arguments:
            embeddings_path_dict_or_list (`Dict[str, str]` or `List[str]`):
                Dictionary of token to embedding path or List of embedding paths to embedding dictionaries.
                The dictionary must have the following keys:
                    - `token`: name of the token to be added to the tokenizers' vocabulary
                    - `embedding`: path to the embedding of the token to be added to the text encoder's embedding matrix
                The list must contain paths to embedding dictionaries where the keys are the tokens and the
                values are the embeddings (same as above dictionary definition).
            allow_replacement (`bool`, *optional*, defaults to `False`):
                Whether to allow replacement of existing tokens in the tokenizer's vocabulary. If `False`
                and a token is already in the vocabulary, an error will be raised.
        Returns:
            None
        """
        # Validate that inheriting class instance contains required attributes
        self._validate_method_call(self.load_textual_inversion_embeddings)

        if isinstance(embedding_path_dict_or_list, dict):
            for token, embedding_path in embedding_path_dict_or_list.items():
                
                embedding_dict = torch.load(embedding_path, map_location=self.text_encoder.device)
                embedding, is_multi_vec_token = self._extract_embedding_from_dict(embedding_dict)

                self._validate_token_update(token, allow_replacement, is_multi_vec_token)
                self.add_textual_inversion_embedding(token, embedding)
        elif isinstance(embedding_path_dict_or_list, list):
            for embedding_path in embedding_path_dict_or_list:
                embedding_dict = torch.load(embedding_path, map_location=self.text_encoder.device)
                token = self._extract_token_from_dict(embedding_dict)
                embedding, is_multi_vec_token = self._extract_embedding_from_dict(embedding_dict)

                self._validate_token_update(token, allow_replacement, is_multi_vec_token)
                self.add_textual_inversion_embedding(token, embedding)
        else:
            raise ValueError(
                f"Type {type(embedding_path_dict_or_list)} is invalid. The value passed to `embedding_path_dict_or_list` "
                "must be a dictionary that maps a token to it's embedding file path "
                "or a list of paths to embedding files containing embedding dictionaries."
            )

    def add_textual_inversion_embedding(self, token: str, embedding: torch.Tensor):
        r"""
        Adds a token to the tokenizer's vocabulary and an embedding to the text encoder's embedding matrix.
        Arguments:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
            embedding (`torch.Tensor`):
                The embedding of the token to be added to the text encoder's embedding matrix
        Returns:
            None
        """
        # NOTE: Not clear to me that we intend for this to be a public/exposed method.
        # Validate that inheriting class instance contains required attributes
        self._validate_method_call(self.load_textual_inversion_embeddings)

        embedding = embedding.to(self.text_encoder.dtype)

        if not isinstance(self.tokenizer, MultiTokenCLIPTokenizer):
            if token in self.tokenizer.get_vocab():
                # If user has allowed replacement and the token exists, we only need to
                # extract the existing id and update the embedding
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
            else:
                # If the token does not exist, we add it to the tokenizer, then resize and update the
                # text encoder acccordingly
                self.tokenizer.add_tokens([token])

                token_id = self.tokenizer.convert_tokens_to_ids(token)
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))
                self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
        else:
            if token in self.tokenizer.token_map:
                # If user has allowed replacement and the token exists, we need to
                # remove all existing tokens associated with the old embbedding and
                # upddate with the new ones
                indices_to_remove = []
                for token_to_remove in self.tokenizer.token_map[token]:
                    indices_to_remove.append(self.tokenizer.get_added_vocab()[token_to_remove])

                    # Remove old  tokens from tokenizer
                    self.tokenizer.added_tokens_encoder.pop(token_to_remove)

                # Convert indices to remove to tensor
                indices_to_remove = torch.LongTensor(indices_to_remove)

                # Remove old tokens from text encoder
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                indices_to_keep = torch.arange(0, token_embeds.shape[0])
                indices_to_keep = indices_to_keep[indices_to_keep != indices_to_remove].squeeze()
                token_embeds = token_embeds[indices_to_keep]

                # Downsize text encoder
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))

                # Remove token from map so MultiTokenCLIPTokenizer doesn't complain
                # on update
                self.tokenizer.token_map.pop(token)

            # Update token with new embedding
            embedding_dims = len(embedding.shape)
            num_vec_per_token = 1 if embedding_dims == 1 else embedding.shape[0]

            self.tokenizer.add_placeholder_tokens(token, num_vec_per_token=num_vec_per_token)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)

            if embedding_dims > 1:
                for i, token_id in enumerate(token_ids):
                    self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding[i]
            else:
                self.text_encoder.get_input_embeddings().weight.data[token_ids] = embedding

    def _extract_embedding_from_dict(self, embedding_dict: Dict[str, str]) -> torch.Tensor:
        r"""
        Extracts the embedding from the embedding dictionary.
        Arguments:
            embedding_dict (`Dict[str, str]`):
                The embedding dictionary loaded from the embedding path
        Returns:
            embedding (`torch.Tensor`):
                The embedding to be added to the text encoder's embedding matrix
            is_multi_vec_token (`bool`):
                Whether the embedding is a multi-vector token or not
        """
        is_multi_vec_token = False
        # auto1111 embedding case
        if "string_to_param" in embedding_dict:
            embedding_dict = embedding_dict["string_to_param"]
            embedding = embedding_dict["*"]
        else:
            embedding = list(embedding_dict.values())[0]

        if len(embedding.shape) > 1:
            # If the embedding has more than one dimension,
            # We need to ensure the tokenizer is a MultiTokenTokenizer
            # because there is branching logic that depends on that class
            if not isinstance(self.tokenizer, MultiTokenCLIPTokenizer):
                raise ValueError(
                    f"{self.__class__.__name__} requires `self.tokenizer` of type `MultiTokenCLIPTokenizer` for loading embeddings with more than one dimension."
                )
            is_multi_vec_token = True

        return embedding, is_multi_vec_token

    def _extract_token_from_dict(self, embedding_dict: Dict[str, str]) -> str:
        r"""
        Extracts the token from the embedding dictionary.
        Arguments:
            embedding_dict (`Dict[str, str]`):
                The embedding dictionary loaded from the embedding path
        Returns:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
        """
        # auto1111 embedding case
        if "string_to_param" in embedding_dict:
            token = embedding_dict["name"]
            return token

        return list(embedding_dict.keys())[0]

    def _validate_method_call(self, method: Callable):
        r"""
        Validates that the method is being called from a class instance that has the required attributes.
        Arguments:
            method (`function`):
                The class's method being called
        Raises:
            ValueError:
                If the method is being called from a class instance that does not have
                the required attributes, the method will not be callable.
        Returns:
            None
        """
        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PreTrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PreTrainedTokenizer` for calling `{method.__name__}`"
            )

        if not hasattr(self, "text_encoder") or not isinstance(self.text_encoder, PreTrainedModel):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` of type `PreTrainedModel` for calling `{method.__name__}`"
            )

    def _validate_token_update(self, token, allow_replacement=False, is_multi_vec_token=False):
        r"""Validates that the token is not already in the tokenizer's vocabulary.
        Arguments:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
            allow_replacement (`bool`):
                Whether to allow replacement of the token if it already exists in the tokenizer's vocabulary
            is_multi_vec_token (`bool`):
                Whether the embedding is a multi-vector token or not
        Raises:
            ValueError:
                If the token is already in the tokenizer's vocabulary and `allow_replacement` is False.
        Returns:
            None
        """
        if (not is_multi_vec_token and token in self.tokenizer.get_vocab()) or (
            is_multi_vec_token and token in self.tokenizer.token_map
        ):
            if allow_replacement:
                print(
                    f"Token {token} already in tokenizer vocabulary. Overwriting existing token and embedding with the new one."
                )
            else:
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name."
                )


class MultiTokenCLIPTokenizer(CLIPTokenizer):
    """Tokenizer for CLIP models that have multi-vector tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_map = {}

    def add_placeholder_tokens(self, placeholder_token, *args, num_vec_per_token=1, **kwargs):
        r"""Adds placeholder tokens to the tokenizer's vocabulary.
        Arguments:
            placeholder_token (`str`):
                The placeholder token to be added to the tokenizers' vocabulary and token map.
            num_vec_per_token (`int`):
                The number of vectors per token. Defaults to 1.
            *args:
                The arguments to be passed to the tokenizer's `add_tokens` method.
            **kwargs:
                The keyword arguments to be passed to the tokenizer's `add_tokens` method.
        Returns:
            None
        """
        output = []
        if num_vec_per_token == 1:
            self.add_tokens(placeholder_token, *args, **kwargs)
            output.append(placeholder_token)
        else:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f"_{i}"
                self.add_tokens(ith_token, *args, **kwargs)
                output.append(ith_token)
        # handle cases where there is a new placeholder token that contains the current placeholder token but is larger
        for token in self.token_map:
            if token in placeholder_token:
                raise ValueError(
                    f"The tokenizer already has placeholder token {token} that can get confused with"
                    f" {placeholder_token}keep placeholder tokens independent"
                )
        self.token_map[placeholder_token] = output

    def replace_placeholder_tokens_in_text(self, text, vector_shuffle=False, prop_tokens_to_load=1.0):
        r"""Replaces placeholder tokens in text with the tokens in the token map.
        Opttionally, implements:
            a) vector shuffling (https://github.com/rinongal/textual_inversion/pull/119)where
            shuffling tokens were found to force the model to learn the concepts more descriptively.
            b) proportional token loading so that not every token in the token map is loaded on each call;
            used as part of progressive token loading during training which can improve generalization
            during inference.
        Arguments:
            text (`str`):
                The text to be processed.
            vector_shuffle (`bool`):
                Whether to shuffle the vectors in the token map. Defaults to False.
            prop_tokens_to_load (`float`):
                The proportion of tokens to load from the token map. Defaults to 1.0.
        Returns:
            `str`: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(self.replace_placeholder_tokens_in_text(text[i], vector_shuffle=vector_shuffle))
            return output
        for placeholder_token in self.token_map:
            if placeholder_token in text:
                tokens = self.token_map[placeholder_token]
                tokens = tokens[: 1 + int(len(tokens) * prop_tokens_to_load)]
                if vector_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, " ".join(tokens))
        return text

    def __call__(self, text, *args, vector_shuffle=False, prop_tokens_to_load=1.0, **kwargs):
        """Wrapper around [`~transformers.tokenization_utils.PreTrainedTokenizerBase.__call__`] method
        but first replace placeholder tokens in text with the tokens in the token map.
        Returns:
            [`~transformers.tokenization_utils_base.BatchEncoding`]
        """
        return super().__call__(
            self.replace_placeholder_tokens_in_text(
                text,
                vector_shuffle=vector_shuffle,
                prop_tokens_to_load=prop_tokens_to_load,
            ),
            *args,
            **kwargs,
        )

    def encode(self, text, *args, vector_shuffle=False, prop_tokens_to_load=1.0, **kwargs):
        """Wrapper around the tokenizer's [`transformers.tokenization_utils.PreTrainedTokenizerBase.encode`] method
        but first replaces placeholder tokens in text with the tokens in the token map.
        Arguments:
            text (`str`):
                The text to be encoded.
            *args:
                The arguments to be passed to the tokenizer's `encode` method.
            vector_shuffle (`bool`):
                Whether to shuffle the vectors in the token map. Defaults to False.
            prop_tokens_to_load (`float`):
                The proportion of tokens to load from the token map. Defaults to 1.0.
            **kwargs:
                The keyword arguments to be passed to the tokenizer's `encode` method.
        Returns:
            List[`int`]: sequence of ids (integer)
        """
        return super().encode(
            self.replace_placeholder_tokens_in_text(
                text,
                vector_shuffle=vector_shuffle,
                prop_tokens_to_load=prop_tokens_to_load,
            ),
            *args,
            **kwargs,
        )
