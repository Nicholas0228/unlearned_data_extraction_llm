import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F    
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import  StoppingCriteriaList
from transformers.utils import is_torchdynamo_compiling


@torch.no_grad()
def get_components(
    model,
    inputs: Optional[torch.Tensor] = None,
    generation_config = None,
    logits_processor = None,
    stopping_criteria = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model = None,
    streamer = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    r"""
    """
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    model._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    generation_config, model_kwargs = model._prepare_generation_config(generation_config, **kwargs)
    model._validate_model_kwargs(model_kwargs.copy())
    try:
        model._validate_assistant(assistant_model)
    except:
        print('no _validate_assistant')

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    try:
        model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
    except:
        print('No _prepare_special_tokens')
    # decoder-only models must use left-padding for batched generation.
    if not model.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        try:
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )
        except:
            print(' GenerationConfig object has no attribute _pad_token_tensor')

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache
    
    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        try:
            model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )
        except:
            model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
                    inputs_tensor)
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = model.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if model._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
        model_kwargs["num_logits_to_keep"] = 1

    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
    cache_name = "past_key_values" if "mamba" not in model.__class__.__name__.lower() else "cache_params"
    user_defined_cache = model_kwargs.get(cache_name)
    max_cache_length = generation_config.max_length
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not model.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    model._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )
    return (prepared_logits_processor, prepared_stopping_criteria, generation_config, synced_gpus, streamer), model_kwargs