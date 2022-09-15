import torch

SOS_token = 0
EOS_token = 1

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<unk>", 1: "<pad>", 2: "<bos>", 3: '<eos>'}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in list(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def get_indices_from_sentence(sentence, lang):
    ten = torch.zeros(len(sentence))
    for i, word in enumerate(sentence):
        ten[i] = lang.word2index[word]
    return ten

def get_index_tensor_from_sentences(sentences, lang):
    index_tensor = torch.tensor([])
    for sentence in sentences:
        index_tensor = 5