from tqdm import tqdm
# from data_module import TextDatasetQA, custom_data_collator, get_batch_loss
from data_module import TextDatasetNoQASet, custom_data_collator, get_batch_loss
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import os, hydra
import evaluate
import json
from pathlib import Path
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml
import torch.nn as nn
import copy
import math
from get_info import get_components
# from merge_models import CustomModelForCausalLM

def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}


        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)


        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)

        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        
        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        eval_logs['average_perturb_loss'] = eval_logs.get('average_perturb_loss', []) + (perturb_loss/num_token_perturb).tolist()
        eval_logs['avg_paraphrased_loss'] = eval_logs.get('avg_paraphrased_loss', []) + (gt_loss/num_token_gt).cpu().numpy().tolist()

        eval_logs['paraphrased_loss'] = eval_logs.get('paraphrased_loss', []) + gt_loss.tolist()
        eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + perturb_loss.tolist()

        eval_logs['num_token_paraphrased'] = eval_logs.get('num_token_paraphrased', []) + num_token_gt.tolist()
        eval_logs['num_token_perturb'] = eval_logs.get('num_token_perturb', []) + num_token_perturb.tolist()

    return eval_logs

def get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key):

    torch_format_dataset = TextDatasetNoQASet( 
            folder, 
            tokenizer=tokenizer, 
            model_family=cfg.model_family, 
            max_length=cfg.generation.max_length, 
            split=split, 
        ) 


    torch_format_dataset.data = torch_format_dataset.data.select(range(len(torch_format_dataset.data)))

    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator)

    return eval_dataloader

def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, pretrained_model=None, gamma=1.0, logsoftmax=True, sample=True, minus_value=None):

    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, pretrained_model=pretrained_model, tokenizer=tokenizer, gamma=gamma, logsoftmax=logsoftmax, sample=sample, minus_value=minus_value)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)

        eval_logs['avg_gt_loss'] = eval_logs.get('avg_gt_loss', []) + (gt_loss/num_token_gt).cpu().numpy().tolist()
        eval_logs['gt_loss'] = eval_logs.get('gt_loss', []) + gt_loss.tolist()
        eval_logs['num_token_gt'] = eval_logs.get('num_token_gt', []) + num_token_gt.tolist()


    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))
    # eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))

    eval_logs['generated_text'] = list(zip(input_strings, gen_outputs,ground_truths))
    return eval_logs

def get_kl_divergence(model, oracle_model, eval_dataloader):
    '''
    Compute the KL divergence of each task on the unlearned model and the oracle model (the fine-tuned model).
    '''
    
    kl_outputs = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            outputs_oracle_model = oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
            
            probs = F.log_softmax(outputs.logits, dim=-1)
            probs_oracle_model = F.log_softmax(outputs_oracle_model.logits, dim=-1)
            kl_divergence = nn.functional.kl_div(probs, probs_oracle_model, reduction='none', log_target=True)
            kl_outputs.extend(kl_divergence.sum(axis=2).mean(axis=1).cpu().numpy().tolist())
    return kl_outputs

@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    print(len(cfg.data_path))
    print(len(cfg.question_key))
    assert len(cfg.data_path)==len(cfg.split_list)==len(cfg.eval_task)==len(cfg.question_key)==len(cfg.answer_key)==len(cfg.base_answer_key)==len(cfg.perturbed_answer_key), "data_path, split, eval_task, question_key, and answer_key must be the same length"
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    # max_length = 500
    batch_size = cfg.batch_size

    model = None
    config = AutoConfig.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", trust_remote_code = True, device_map=device_map)
    for attempt in range(1):
        try:
        # do thing
            if cfg.use_pretrained:
                print(f"Loading pretrained checkpoint from {model_id}")
                model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
        except:

            print('Now try customized one')
            if cfg.use_pretrained:
                print(f"Loading checkpoint from {model_id}")
                model = CustomModelForCausalLM.from_pretrained(model_id, device_map=device_map)
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                model = CustomModelForCausalLM.from_pretrained(cfg.model_path, device_map=device_map)
            break
            
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")


    pretrained_model = None
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    config = AutoConfig.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", trust_remote_code = True, device_map=device_map)
    for attempt in range(3):
        if cfg.pretrained_path is not None:
            try:

                print(f"Loading pretrained from {cfg.pretrained_path}")
                pretrained_model = AutoModelForCausalLM.from_pretrained(cfg.pretrained_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            except Exception as e:
                print(e)
                continue
            # perhaps reconnect, etc.
            else:
                break
        else:
            try:

                print(f"Loading pretrained from {model_id}")
                pretrained_model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            except Exception as e:
                print(e)
                continue
            # perhaps reconnect, etc.
            else:
                break
    else:
        print("Error: could not load model")
    
    def reinitialize_weights(model) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    minus_value = cfg.minus_value
    for i, (folder, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key, gamma, logsoftmax, sample) in enumerate(zip(cfg.data_path,  cfg.question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key, cfg.gamma_list, cfg.logsoftmax_list, cfg.sample_list)):
        split = cfg.split
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print('current info')
        print(i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key, gamma, logsoftmax, sample) )
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}_{logsoftmax}_{gamma}_Sample_{sample}_{minus_value}.json")
        save_filename = save_filename if world_size == 1 else os.path.join(cfg.save_dir, f"{split}_{eval_task}_{logsoftmax}_{gamma}_Sample_{sample}_{os.environ.get('LOCAL_RANK', '0')}.json")
        
        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue
       
        eval_dataloader = get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
        print('begin eval', save_filename)
        eval_logs = get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, pretrained_model=pretrained_model, gamma=gamma, logsoftmax=logsoftmax, sample=sample, minus_value=minus_value)

        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, pretrained_model, tokenizer, gamma, logsoftmax, sample,minus_value=None):

    input_ids = batch["input_ids"]
    ground_truth = []
    input_strings = []
    input_string_decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    for s in input_string_decoded:
        word_count = len(s.split())
        
        # Find the index of the middle word
        middle_index = word_count // 2
        
        # Split the string into two halves based on the middle index
        input_string = ' '.join(s.split()[:middle_index])  # First half
        ground_truth_i = ' '.join(s.split()[middle_index:])  # Second half
        
        # Print the input_string and ground_truth_i
        ground_truth.append(ground_truth_i)
        input_strings.append(input_string)
    # ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    # input_strings = [s.split(split_symbol)[0] for s in input_strings]

    #add ["/INST "] to the end of each string
    # if cfg.model_family == 'llama2-7b':
    #     input_strings = [s + split_symbol for s in input_strings]
        
    #we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]
    
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id


    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    
    #now generate
    torch.manual_seed(0)

    out = contrasting_generation(model, inputs, cfg, left_pad_tokenizer, tokenizer, pretrained_model=pretrained_model, gamma=gamma, logsoftmax=logsoftmax, sample=sample, minus_value=minus_value)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    return input_strings, strs, ground_truth



def contrasting_generation(model, inputs, cfg, left_pad_tokenizer, tokenizer, pretrained_model=None, gamma=1.0, logsoftmax=True, sample=True, minus_value=None):
    input_ids = inputs.input_ids

    gen_num = input_ids.shape[0]
    max_token = cfg.generation.max_length

    get_result = torch.zeros([gen_num,max_token-inputs.input_ids.shape[-1]], dtype=torch.int64).cuda()
    current_input = copy.deepcopy(inputs)

    # params, model_kwargs = model.get_logits_processor(inputs.input_ids, 
    #                     attention_mask=inputs.attention_mask,
    #                     max_length=cfg.generation.max_length, 
    #                     max_new_tokens=cfg.generation.max_new_tokens, 
    #                     do_sample=False,
    #                    use_cache=True,  
    #                     pad_token_id=left_pad_tokenizer.eos_token_id)
    params, model_kwargs = get_components(model, inputs.input_ids, 
                        attention_mask=inputs.attention_mask,
                        max_length=cfg.generation.max_length, 
                        max_new_tokens=cfg.generation.max_new_tokens, 
                        do_sample=False,
                       use_cache=True,  
                        pad_token_id=left_pad_tokenizer.eos_token_id)

    if pretrained_model is not None:
        params1, model_kwargs1 = get_components(pretrained_model, inputs.input_ids, 
                            attention_mask=inputs.attention_mask,
                            max_length=cfg.generation.max_length, 
                            max_new_tokens=cfg.generation.max_new_tokens, 
                            do_sample=False,
                        use_cache=True,  
                            pad_token_id=left_pad_tokenizer.eos_token_id)

    logits_processor, stopping_criteria, generation_config, synced_gpus, streamer = params
    def compare_model_inputs(dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1:
            tensor1 = dict1[key]
            tensor2 = dict2[key]
            print(tensor1, tensor2)
            if not torch.equal(tensor1, tensor2):
                return False
        return True
    print(f'Using our implementation for {logsoftmax} {gamma}')
    update_flag = torch.ones(gen_num, dtype=torch.bool).cuda()  # bool 类型，初始为 True
    input_ids = copy.deepcopy(inputs.input_ids)
    model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)
    if pretrained_model is not None:
        model_kwargs1 = pretrained_model._get_initial_cache_position(input_ids, model_kwargs1)
    torch.manual_seed(0)
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    max_length = generation_config.max_length
    output_scores = generation_config.output_scores
    return_dict_in_generate = generation_config.return_dict_in_generate
    scores = () if (return_dict_in_generate and output_scores) else None
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    while model._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
    # for k in range (max_token-input_ids.shape[-1]):
        
        model_inputs=model.prepare_inputs_id(input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer , **model_kwargs)
        if pretrained_model is not None:
            model_inputs_pre=pretrained_model.prepare_inputs_id(input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer , **model_kwargs1)
        # print(model_inputs)
        # logits0 = model(current_input.input_ids, attention_mask=current_input.attention_mask, use_cache=True, output_hidden_states=False, return_dict=True,).logits

        outputs0 = model(**model_inputs, return_dict=True)
        logits0 = outputs0.logits[:, -1, :].float()
        if pretrained_model is not None:
            outputs1 = pretrained_model(**model_inputs_pre, return_dict=True)
            logits1 = outputs1.logits[:, -1, :].float()
        else:
            outputs1 = outputs0
            logits1 = outputs1.logits[:, -1, :].float()
        if gamma < 0:
            logits0, logits1 = logits1, logits0
 
        if logsoftmax:
            
            logits0=F.log_softmax(logits0, dim=-1)
            logits1=F.log_softmax(logits1, dim=-1)
            logits = (1-abs(gamma)) * logits1 + abs(gamma) * logits0
            logits = torch.exp(logits)
        else:
            logits = (1-abs(gamma)) * logits1 + abs(gamma) * logits0
        # next_token_logits = logits.clone()
        next_token_logits = logits

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        # print(next_token_scores)
        if sample:
            if logsoftmax:
                probs = next_token_scores
            else:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            if minus_value is None:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            else:
                max_logits0 = torch.max(logits0, dim=-1)[0]
                # 找到logits0中大于max_logits0的条件下的索引
                mask = logits0 > (max_logits0.unsqueeze(-1)-minus_value)

                # 使用掩码在logits中选择对应的最大值
                logits_masked = logits.masked_fill(~mask, float('-inf'))  # 使用 -inf 填充不符合条件的位置
                next_tokens = torch.argmax(logits_masked, dim=-1).cuda()
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs0,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        if pretrained_model is not None:
            model_kwargs1 = pretrained_model._update_model_kwargs_for_generation(
                outputs1,
                model_kwargs1,
                is_encoder_decoder=pretrained_model.config.is_encoder_decoder,
            )
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs0, outputs1



    return input_ids



def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_rouge_recall(gen_outputs, ground_truths):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores['rouge1'].recall)
        rougeL_recall.append(rouge_scores['rougeL'].recall)


    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}

if __name__ == "__main__":
    main()

