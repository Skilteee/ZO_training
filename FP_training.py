from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments
import os
import transformers
import torch
import re
from datautils import get_loaders
from FP_zo_trainer import ZOTrainer
import numpy as np
import random

model_name = "facebook/opt-2.7b"
model_family = re.findall(r"/(.*?)-", model_name)[0]
model_nick_name = model_name.split("/")[-1]
is_llama = 'llama' in model_name.lower()
calib_dataset = 'wikitext2'
cache_dir = './cache'
nsamples = 1280
seed = 4
lr = 1e-7
zo_eps = 1e-4

def set_seed(seed):
    # Python 内置随机数
    random.seed(seed)

    # Numpy 随机数
    np.random.seed(seed)

set_seed(seed)

if is_llama:
    model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 # device_map="auto",
                                                          )
else:
    model = transformers.OPTForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 # device_map="auto",
                                                 )
model = model.cuda()

model.seqlen = 128
cache_dataloader = f'{cache_dir}/dataloader_{model_family}_{calib_dataset}_{nsamples}_{model.seqlen}_{seed}.cache'
if os.path.exists(cache_dataloader):
    dataloader = torch.load(cache_dataloader)
else:
    dataloader, _ = get_loaders(
        calib_dataset,
        nsamples=nsamples,
        seed=seed,
        model=model_name,
        seqlen=model.seqlen,
    )
    torch.save(dataloader, cache_dataloader)

all_input_ids = []
all_attention_masks = []

for input_ids, attention_mask in dataloader:
    all_input_ids.extend(input_ids.tolist())
    all_attention_masks.extend(torch.ones(input_ids.shape, dtype=torch.long).tolist())

hf_dataset = Dataset.from_dict({
    "input_ids": all_input_ids,
    "attention_mask": all_attention_masks
})

bs = 4
named_parameters_to_optim = []

for name, param in model.named_parameters():
    param.requires_grad = False
    if 'bias' not in name and 'norm' not in name:
        named_parameters_to_optim.append((name, param))

named_parameters_to_optim = named_parameters_to_optim[1:-1] if is_llama else named_parameters_to_optim[2:]
zo_trainer = ZOTrainer(
    named_parameters_to_optim=named_parameters_to_optim,
    lr=lr,
    zo_eps=zo_eps,
    is_llama=is_llama
)

log_iter = 20
for epoch in range(10):
    loss_list = []
    for i in range(0, len(hf_dataset), bs):
        batch = hf_dataset[i:i + bs]
        batch['input_ids'] = torch.tensor(batch['input_ids'], dtype=torch.long, device='cuda')
        batch['attention_mask'] = torch.tensor(batch['attention_mask'], dtype=torch.long, device='cuda')

        loss1 = zo_trainer.zo_step(model, batch)
        zo_trainer.zo_update()
        loss_list.append(loss1.item())

        if i % log_iter == 0:
            print(f"Step {i}, Loss: {np.mean(loss_list)}")




