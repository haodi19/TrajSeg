#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        # self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,    # TODO: ?
        return_dict: Optional[bool] = None,
        trajectories: Optional[torch.FloatTensor] = None

    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 这个函数对应LLaMA-VID的函数，但LISA只有推理的时候直接执行该函数(训练也执行，但是被封装在一个大的forward里了)
        # 另外LLaMA-VID中use_cache为True, LISA为False, 即LISA不使用past_key_values, 每次input_ids都是完整的
        # 语言模型多次循环生成token，第一次循环input_ids：
        # input_ids:tensor([[    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
        #                    21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
        #                      322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
        #                    29889,  3148,  1001, 29901, 32001,  -200, 32002, 22172,   319,  1799,
        #                     9047, 13566, 29901]], device='cuda:0')
        # attention_mask: None
        # use_cache: False
        # output_attentions: False
        # output_hidden_states: True
        # images: torch.Size([bs, 3, 224, 224]) -> torch.Size([bs, video_len, 3, 224, 224])
        # return_dict: True
        # 其余为None
        
        # output_attentions: False
        # output_hidden_states: True
        # return_dict: True
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            trajectory_features,
            mlvl_features
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, trajectories
        )
        
        # 这里prepare_inputs_labels_for_multimodal和LLaMA-VID功能类似，只是因use_cache的值导致输出不同
        # input_ids: None
        # inputs_embeds: torch.Size([1, 298, 4096]), torch.Size([1, 299, 4096])...
        # past_key_values: None
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        # self.model: <class 'model.LISA.LisaModel'>
        # 实际执行transformers/models/llama/modeling_llama.py LlamaModel.forward()
        # 这里self.model就是纯语言模型了，输入token即可(不再区分视觉token与文本token, 上面一步已经将他们合并)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # outputs.keys(): odict_keys(['last_hidden_state', 'hidden_states'])
        # outputs['last_hidden_state']:torch.Size([1, 298, 4096])
        # outputs['hidden_states']: tuple, len=33, outputs['hidden_states'][0]:torch.Size([1, 298, 4096])
        
        # hidden_states: torch.Size([1, 298, 4096]), 这里hidden_states就是outputs['last_hidden_state']
        # logits: torch.Size([1, 298, 32004])

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if trajectory_features is not None and labels is not None:
            loss += torch.stack(trajectory_features).sum() * 0.0
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        trajectory = None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
                "trajectories": trajectory
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
