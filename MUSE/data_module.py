import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import os
from utils import get_model_identifiers_from_yaml


def convert_src_data_to_model_format(tokenizer, max_length, answer, model_configs):
    full_text =  answer

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


class TextForgetDatasetNoQASet(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10"):
        super(TextForgetDatasetNoQASet, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.split1, self.split2 = "forget", "retain"


    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            
            torch.manual_seed(idx)
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            answer = data[idx]['text']
                
            converted_data = convert_src_data_to_model_format(self.tokenizer, self.max_length, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextDatasetNoQASet(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, data_key='text'):
        super(TextDatasetNoQASet, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.data_key = data_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        answers = self.data[idx][self.data_key]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_src_data_to_model_format(self.tokenizer, self.max_length, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()



# class TextDatasetNoQAForgetSet(Dataset):
#     def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, data_key='text'):
#         super(TextDatasetNoQAForgetSet, self).__init__()
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#         self.data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']

#         self.model_configs = get_model_identifiers_from_yaml(model_family)
#         self.data_key = data_key

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         answers = self.data[idx][self.data_key]

#         if isinstance(answers, str):
#             answers = [answers]

#         pad_input_ids_list = []
#         label_list = []
#         pad_attention_mask_list = []

#         for answer in answers:
#             converted_data = convert_src_data_to_model_format(self.tokenizer, self.max_length, answer, self.model_configs)
#             pad_input_ids_list.append(converted_data[0])
#             label_list.append(converted_data[1])
#             pad_attention_mask_list.append(converted_data[2])

#         return torch.stack(pad_input_ids_list).squeeze(),\
#                 torch.stack(label_list).squeeze(),\
#                 torch.stack(pad_attention_mask_list).squeeze()


class TextDatasetNoQAForgetSet(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, data_key='text'):
        super(TextDatasetNoQAForgetSet, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # self.data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']
        self.forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split+'.json'))['train']
        retain_split = "full_minus_" + split 
        self.retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split+'.json'))['train']
        
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.data_key = data_key
        self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            
            torch.manual_seed(idx)
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            answer = data[idx][self.data_key]
                
            converted_data = convert_src_data_to_model_format(self.tokenizer, self.max_length, answer, self.model_configs)
            rets.append(converted_data)
        return rets
            

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_forget(samples):
    rets = []
    if len(samples[0]) == 3:
        idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
        data_types = ["idk", "forget", "retain"]
    elif len(samples[0]) == 2:
        forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
        data_types = ["forget", "retain"]
    for data_type in data_types:
        if data_type == "forget":
            data = forget_samples 
        elif data_type == "retain":
            data = retain_samples 
        elif data_type == "idk":
            data = idk_samples
         
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))

    return rets


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
