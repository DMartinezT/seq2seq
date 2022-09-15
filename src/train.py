
from src.lang import Lang
import src.utils as utils
from src.transformer import Seq2SeqTransformer
from src.transformer import create_mask
from src.transformer import generate_square_subsequent_mask
from src.transformer import FactorExpansionDataset


import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer

from torchinfo import summary

import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']



factors_lang = Lang("factors_lang")
expansions_lang = Lang("expansions_lang")



def set_langs(factors, expansions):
    print("Device: ", DEVICE)
    for factor in factors:
        for sentence in factor:
            factors_lang.addSentence(sentence)
    for expansion in expansions:
        for sentence in expansion:
            expansions_lang.addSentence(sentence)
    return


def token_transform_factors(sentence):
    tokenized_sentence = torch.zeros(len(sentence), dtype = torch.long)
    for i, word in enumerate(sentence):
        tokenized_sentence[i] = factors_lang.word2index[word]    
    return tokenized_sentence

def token_transform_expansions(sentence):
    tokenized_sentence = torch.zeros(len(sentence), dtype = torch.long)
    for i, word in enumerate(sentence):
        tokenized_sentence[i] = expansions_lang.word2index[word]    
    return tokenized_sentence

def word_transform_expansions(indices):
    sentence = ''
    for i in indices:
        sentence = sentence + expansions_lang.index2word[i]
    return sentence

text_transform_factors = utils.sequential_transforms(token_transform_factors, utils.tensor_transform)
text_transform_expansions = utils.sequential_transforms(token_transform_expansions, utils.tensor_transform)

# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform_factors(src_sample))
        tgt_batch.append(text_transform_expansions(tgt_sample))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch




def train_epoch(model, optimizer, train_dataloader, loss_fn):
    model.train()
    losses = 0
    

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform_factors(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()

    return word_transform_expansions(list(tgt_tokens.cpu().numpy())).replace("<bos>", "").replace("<eos>", "")









def train(factors, expansions, p = 0.7, NUM_EPOCHS = 16):
    torch.manual_seed(0)

    SRC_VOCAB_SIZE = factors_lang.n_words
    TGT_VOCAB_SIZE = expansions_lang.n_words
    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 256
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    PATH = '../models/'

    n = len(factors)
    threshold = int(n*p)
    train_iter = FactorExpansionDataset(factors[:threshold], expansions[:threshold])
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_iter = FactorExpansionDataset(factors[threshold+1:], expansions[threshold+1:])
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, val_dataloader, loss_fn)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    

    with open(PATH + 'params.txt', 'w') as f:
        f.write(str(SRC_VOCAB_SIZE) + '\n')
        f.write(str(TGT_VOCAB_SIZE)+ '\n')
        f.write(str(EMB_SIZE)+ '\n')
        f.write(str(NHEAD)+ '\n')
        f.write(str(FFN_HID_DIM)+ '\n')
        f.write(str(BATCH_SIZE)+ '\n')
        f.write(str(NUM_ENCODER_LAYERS)+ '\n')
        f.write(str(NUM_DECODER_LAYERS)+ '\n')
    torch.save(transformer.state_dict(), PATH + 'transformer.pt')



def read_transfomer(PATH = '../models/'):

    l = list()
    with open(PATH + 'params.txt', 'r') as file:
        for f in file:
            l.append(f.strip())

    SRC_VOCAB_SIZE = int(l[0])
    TGT_VOCAB_SIZE = int(l[1])
    EMB_SIZE = int(l[2])
    NHEAD = int(l[3])
    FFN_HID_DIM = int(l[4])
    BATCH_SIZE = int(l[5])
    NUM_ENCODER_LAYERS = int(l[6])
    NUM_DECODER_LAYERS = int(l[7])

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    transformer.load_state_dict(torch.load(PATH + 'transformer.pt'))
        
    return transformer



def read_aux(goal_translate, goal_target, PATH = '../models/'):

    l = list()
    with open(PATH + 'params.txt', 'r') as file:
        for f in file:
            l.append(f.strip())

    SRC_VOCAB_SIZE = int(l[0])
    TGT_VOCAB_SIZE = int(l[1])
    EMB_SIZE = int(l[2])
    NHEAD = int(l[3])
    FFN_HID_DIM = int(l[4])
    BATCH_SIZE = int(l[5])
    NUM_ENCODER_LAYERS = int(l[6])
    NUM_DECODER_LAYERS = int(l[7])

    

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    transformer.load_state_dict(torch.load(PATH + 'transformer.pt'))

    for sentence in goal_translate:
        print('*************************************************************')
        print('Factors: ', sentence)
        print('Expansions: ', translate(transformer, sentence))

    iter = FactorExpansionDataset(goal_translate, goal_target)
    batch_size = 2
    train_dataloader = DataLoader(iter, batch_size=batch_size, collate_fn=collate_fn) 
    for i, d in enumerate(train_dataloader):
        if i == 1:
            break
        src, tgt = d
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        summary(transformer, input_data=[src, tgt_input, src_mask, tgt_mask, 
        src_padding_mask, tgt_padding_mask, src_padding_mask])
