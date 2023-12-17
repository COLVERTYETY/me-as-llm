# %%
import os
import time
import math
import pickle
from contextlib import nullcontext

from model import llm, Mllm
from BPE import BPE_Tokenizer

import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# %%
BLOCK_SIZE=128
BATCH_SIZE=64
model_folder = "highlr"

# %%
# create dataset split files
# check if the dataset is already split

if os.path.exists('data/train.txt') and os.path.exists('data/test.txt'):
    print('Dataset split files already exist')
else:
    print('Dataset split files do not exist')
    print('Creating dataset split files...')
    with open('data/Harry_Potter_all_books_preprocessed.txt', 'r') as f:
        full = f.read()
        train = full[:int(len(full)*0.8)]
        test = full[int(len(full)*0.8):]
        print('Train size:', len(train))
        print('Test size:', len(test))
    with open('data/train.txt', 'w') as f:
        f.write(train)
    with open('data/test.txt', 'w') as f:
        f.write(test)
    print('Done')


# %%
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, block_size, divider=2):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        self.divider = divider
        with open(data_path, 'r') as f:
            full_text = f.read()
            full_text = full_text.replace('.', ' . ')
            full_text = full_text.replace(',', ' , ')
            full_text = full_text.replace('!', ' ! ')
            full_text = full_text.replace('?', ' ? ')
            full_text = full_text.replace(':', ' : ')
            full_text = full_text.replace(';', ' ; ')
            full_text = full_text.replace(')', '')
            full_text = full_text.replace('(', '')
            full_text = full_text.replace('@', '')
            full_text = full_text.replace('|', '')
            full_text = full_text.replace(']', '')
            full_text = full_text.replace('[', '')
            full_text = full_text.replace('~', '')
            full_text = full_text.replace('^', '')
            full_text = full_text.replace('<', '')
            full_text = full_text.replace('>', '')
            full_text = full_text.replace('&', '')
            full_text = full_text.replace('{', '')
            full_text = full_text.replace('}', '')
            full_text = full_text.replace('+', '')
            # full_text = full_text.replace('-', '')
            full_text = full_text.replace('tititi', '')
            full_text = full_text.replace('orerer', '')
            full_text = full_text.replace('errero', '')
            full_text = full_text.replace('\u007f', '')
            full_text = full_text.replace('_', '')
            full_text = full_text.replace('%', '')
            full_text = full_text.replace('$', '')
            full_text = full_text.replace('\\', '')
            full_text = full_text.replace('=', '')
            full_text = full_text.replace('#', '')
            full_text = full_text.replace(';', '')
            full_text = full_text.replace(':', '')
            full_text = full_text.encode("ascii", errors="ignore").decode()
            # self.data = full_text.split()
            tokenized = tokenizer.tokenize(full_text)
            # break into blocks
            for i in range(0, len(tokenized) - block_size + 1, block_size//divider):
                self.data.append(tokenized[i:i+block_size])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx][:self.block_size-1], dtype=torch.long)
        Y = torch.tensor(self.data[idx][1:self.block_size], dtype=torch.long)
        return X, Y


# create tokenizer
tokenizer = BPE_Tokenizer()
tokenizer.load_vocab('openweb+Nicolas.json')

# create dataset
print('Creating dataset...')
print('loading Train')
train_set = TokenDataset('data/openwebTrain.txt', tokenizer, BLOCK_SIZE, divider=2)
print('loading Test')
test_set = TokenDataset('data/openwebTest.txt', tokenizer, BLOCK_SIZE,divider=2)

print('loading Tune Train')
tune_train_set = TokenDataset('data/nicolasSTASTrain.txt', tokenizer, BLOCK_SIZE, divider=4)
print('loading Tune Test')
tune_test_set = TokenDataset('data/nicolasSTASTest.txt', tokenizer, BLOCK_SIZE, divider=4)

print('sample from train set:', len(train_set[0]), tokenizer.detokenize(train_set[0][0].tolist()))
print('sample from test set:', len(test_set[0]), tokenizer.detokenize(test_set[0][0].tolist()))

# create dataloader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

tune_train_loader = DataLoader(tune_train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
tune_test_loader = DataLoader(tune_test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

#  check that dataloader works
for batch in train_loader:
    print(batch[0].shape, batch[1].shape)
    break


# %%
VOCAB_SIZE = tokenizer.get_vocab_size()
VOCAB_SIZE

# %%
#  init model
# N_EMBED = VOCAB_SIZE*2
N_EMBED = 256
N_HEAD = 4
N_LAYERS = 6
# model = llm(VOCAB_SIZE, BLOCK_SIZE, N_EMBED, N_HEAD, N_LAYERS)
model = Mllm(VOCAB_SIZE, BLOCK_SIZE, N_EMBED, N_HEAD, N_LAYERS)
model

# %%

def speak(model, tokenizer, prompt, max_len=100, temperature=0.5, repetition = 1.1):
    input_ = tokenizer.tokenize(prompt)
    gen = model.generate(torch.tensor(input_).unsqueeze(0).to(device), max_len, temperature=temperature, repetition_penalty=repetition, top_k=10)
    text = tokenizer.detokenize(gen.cpu().squeeze(0).tolist()) 
    text = text.replace(' . ', '.\n')
    print(text)

# %%
@torch.no_grad()
def estimate_loss(testlodaer, trainloader):
    model.eval()
    # losses = torch.zeros((BATCH_SIZE,BLOCK_SIZE))
    test_losses = []
    for k, (X,Y) in enumerate(tqdm(testlodaer, desc='eval test')):
        X,Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        test_losses.append(loss.item())
    train_losses = []
    for k, (X,Y) in enumerate(tqdm(trainloader, desc='eval train')):
        X,Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        train_losses.append(loss.item())
    model.train()
    return {'test': np.array(test_losses).mean(), 'train':np.array(train_losses).mean()}

# estimate_loss()

# %%
WEIGHT_DECAY = 0.001
LEARNING_RATE = 4e-4
MIN_LR = 9e-5
WARMUP_STEPS = 2
LR_DECAY_ITERS = 25
MAX_ITERS = 60
# LEARNING_RATE = 1e-3
beta1 = 0.9
beta2 = 0.95

# %%
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for WARMUP_STEPS steps
    if it < WARMUP_STEPS:
        return LEARNING_RATE * it / WARMUP_STEPS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > LR_DECAY_ITERS:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_STEPS) / (LR_DECAY_ITERS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

# %%
# optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (beta1, beta2),'cuda')
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(beta1, beta2), weight_decay=WEIGHT_DECAY)
print(optimizer)

# %%
# # compile model
print('Compiling model... (takes a while)')
# model = model.to('cuda')
model = torch.compile(model).to(device)
# # model.train()
print(model)

# %%
writer = SummaryWriter()
# writer.add_graph(model, torch.zeros((BATCH_SIZE, BLOCK_SIZE)).to(device))

best_loss = np.inf

for e in range(MAX_ITERS):
    losses = estimate_loss(test_loader, train_loader)
    print('Epoch', e, 'Losses:', losses)
    speak(model, tokenizer, "This morning on radio news", max_len=256)
    lr = get_lr(e)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if losses['test'] < best_loss:
        best_loss = losses['test']
        torch.save(model.state_dict(), f'{model_folder}/best_llm.pt')
        print('Saved model')
    writer.add_scalar('loss/train', losses['train'], e)
    writer.add_scalar('loss/test', losses['test'], e)
    writer.add_scalar('lr', lr, e)
    for k, (X,Y) in enumerate(tqdm(train_loader, desc='train')):
        X,Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


# %%
model.load_state_dict(torch.load(f'{model_folder}/best_llm.pt'))
model.eval()
speak(model, tokenizer, "They are searching for ", max_len=2048)
model.train()

# %%
#  tune 
LEARNING_RATE = 1e-6
MIN_LR = 1e-7
WARMUP_STEPS = 10
LR_DECAY_ITERS = 100
MAX_ITERS = 1000

# %%

# writer = SummaryWriter()
# writer.add_graph(model, torch.zeros((BATCH_SIZE, BLOCK_SIZE)).cuda())

best_loss = np.inf

for e in range(MAX_ITERS):
    losses = estimate_loss(tune_test_loader, tune_train_loader)
    print('Epoch', e, 'Losses:', losses)
    speak(model, tokenizer, "Hi there! I'm Nicolas", max_len=50)
    lr = get_lr(e)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if losses['test'] < best_loss:
        best_loss = losses['test']
        torch.save(model.state_dict(), f'{model_folder}/best_tune_llm.pt')
        print('Saved model')
    if e % 10 == 0:
        torch.save(model.state_dict(), f'{model_folder}/tune_llm_{e}.pt')
        print('Saved model')
    writer.add_scalar('loss/train', losses['train'], e)
    writer.add_scalar('loss/test', losses['test'], e)
    writer.add_scalar('lr', lr, e)
    for k, (X,Y) in enumerate(tqdm(tune_train_loader, desc='train')):
        X,Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


# %%
# model.load_state_dict(torch.load('small/best_tune_llm.pt'))
# modle = torch.compile(model).to('cuda')
model.load_state_dict(torch.load(f'{model_folder}/tune_llm_990.pt'))
model.eval()
speak(model, tokenizer, "Hi there! I'm Nicolas STAS", max_len=1024, temperature=1.0)

# %%
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def repair_checkpoint(path, out_path="fixed.pt"):
    ckpt = torch.load(path)
    # print(ckpt.keys())
    # in_state_dict = ckpt["model_state_dict"]
    in_state_dict = ckpt
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return  # Do not write checkpoint if no need to repair!
    out_state_dict = {}
    for src_key, dest_key in pairings:
        print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    # ckpt["model_state_dict"] = out_state_dict
    torch.save(out_state_dict, out_path)

# repair_checkpoint('small/best_tune_llm.pt', 'small/best_tune_llm_fixed.pt')
repair_checkpoint(f'{model_folder}/tune_llm_190.pt', f'{model_folder}/tune_llm_190_fixed.pt')

model = Mllm(VOCAB_SIZE, BLOCK_SIZE, N_EMBED, N_HEAD, N_LAYERS)
model.load_state_dict(torch.load(f'{model_folder}/tune_llm_190_fixed.pt'))

# %%
# dynamic quantization
# model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
speak(model.to(device), tokenizer, "Hi there! I'm Nicolas STAS ", max_len=1024, temperature=0.9)
print(model)

# %%
# export model to onnx

dummy_input = torch.zeros((1, BLOCK_SIZE), dtype=torch.int32).to('cuda')
model = model.eval().to('cuda')
# model = torch.jit.script(model)
# onnx_prog = torch.onnx.dynamo_export(model, dummy_input)
torch.onnx.export(model, dummy_input, f'{model_folder}/llm.onnx', opset_version=17, input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'batch_size', 1:'sequence'}, 'output':{0:'batch_size', 1:'sequence'}})

# %%
# ! pip install onnxruntime onnx
# import onnx
from onnxruntime.quantization import quantize_dynamic, preprocess

model_fp32 = f'{model_folder}/llm.onnx'
model_quant = f'{model_folder}/llm.quant.onnx'
preprocess_model = f'{model_folder}/llm.preprocess.onnx'
# preprocess(model_fp32, model_quant, num_bits=8)
# preprocess.quant_pre_process(model_fp32, preprocess_model)

quantized_model = quantize_dynamic(model_fp32, model_quant)

# %%
#  generate with onnx
import onnxruntime
import numpy as np
import time
from collections import Counter

sess_options = onnxruntime.SessionOptions()
sess_options.enable_profiling = True
sess = onnxruntime.InferenceSession(model_quant)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(input_name, label_name)

def genearte_onnx(sess, input_name, label_name, prompt, max_len=100, temperature=0.5, repetition = 1.1, device = 'cuda'):
    input_ = tokenizer.tokenize(prompt)
    input_ = torch.tensor(input_, dtype=torch.int32).unsqueeze(0).to(device)
    gen = []
    for _ in range(max_len):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = input_ if input_.size(1) <= BLOCK_SIZE else input_[:, -BLOCK_SIZE:]
        # forward the model to get the logits for the index in the sequence
        logits = sess.run([label_name], {input_name: idx_cond.cpu().numpy()})[0]
        logits = torch.tensor(logits, dtype=torch.float32).to(device)
        logits = logits[:, -1, :] / temperature
        #  count token repetitions and apply repetition penalty
        counts = Counter(input_.view(-1).tolist())
        # avg = sum(counts.values()) / len(counts)
        for k,v in counts.items():
            logits[:,k] /= repetition**(v)
        # apply softmax to convert logits to (normalized) probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        #  cast to int32
        idx_next = idx_next.to(torch.int32)
        # append sampled index to the running sequence and continue
        input_ = torch.cat((input_, idx_next), dim=1)
        gen.append(idx_next.item())
    return gen

def speak_onnx(sess, input_name, label_name, prompt, max_len=100, temperature=0.5, repetition = 1.1, device = 'cuda'):
    gen = genearte_onnx(sess, input_name, label_name, prompt, max_len, temperature, repetition, device)
    text = tokenizer.detokenize(gen)
    text = text.replace(' . ', '\n')
    print(prompt, text)

speak_onnx(sess, input_name, label_name, "Hi there! I'm Nicolas STAS ", max_len=2048)


