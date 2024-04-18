# transformer.py

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
#from tqdm import tqdm


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)

# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        #print("X in Transformer init")
        super(Transformer, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, d_model) 
        self.positional_encoding = PositionalEncoding(d_model,num_positions,True)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model,num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)


    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        #print("X in Transformer forward")
        output = self.input_embeddings(indices.unsqueeze(0))
        output = self.positional_encoding(output)
        self_attns = []
        self_attns.clear()
        for transformer_layer in self.transformer_layers:
            output, self_attn_scores = transformer_layer(output)
            self_attns.append(self_attn_scores)
        output = self.output_layer(output)
        #print("Output shape before log_softmax", output.shape)
        output = self.log_softmax(output)
        return output[0], self_attn_scores[0]

#d_model is embedding size
#seq_length is always 20 characters
#vocab size is 27
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads: int=2):
        #print("X in MHA init")
        super(MultiHeadAttention, self).__init__()
        #size of each character's projected embedded space
        self.d_model = d_model
        self.num_heads = num_heads
        #key space for a particular head
        self.d_k = d_model // num_heads
        self.non_linear_activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.weight_query =  nn.Linear(d_model,d_model)
        self.weight_key =  nn.Linear(d_model,d_model)
        self.weight_value =  nn.Linear(d_model,d_model)
        self.weight_linear =  nn.Linear(d_model,d_model)
    
    def attention(self,Query,Key,Value):
        #print("X in MHA ATTENTION")
        scores = torch.div(torch.matmul(Query, Key.transpose(-2,-1)) , np.sqrt(self.d_k))
        probs = self.softmax(scores)
        hidden_layer = torch.matmul(probs, Value)
        return hidden_layer, scores
    
    #ref: https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch to understand how to implement multi head attention efficiently.
    def split_heads(self, x):
        #each character is embedded into d_model sized embedding space
        #each sequence consists of 20 characters
        #for each batch
        #print("X in split heads", x.size())
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)
    
    def combine_heads(self, x):
        #print("X in mha combine heads")
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self,Query,Key,Value):
        #print("X in MHA forward")
        Query = self.split_heads(self.non_linear_activation(self.weight_query(Query)))
        Key = self.split_heads(self.non_linear_activation(self.weight_key(Key)))
        Value = self.split_heads(self.non_linear_activation(self.weight_value(Value)))

        attention_op,scores = self.attention(Query,Key,Value)
        output = self.non_linear_activation(self.weight_linear(self.combine_heads(attention_op)))

        return output,scores

class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_internal):
        #print("X in FF init")
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_internal)
        self.linear_2 = nn.Linear(d_internal, d_model)
        self.non_linear_activation = nn.ReLU()

    def forward(self,x):
        #print("X in FF forward")
        x = self.linear_1(x)
        x = self.non_linear_activation(x)
        x = self.linear_2(x)
        return x



# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        #print("X in Transformer Layer init")
        super(TransformerLayer, self).__init__()
        self.attention_layer = MultiHeadAttention(d_model,4)
        self.feed_forward = FeedForward(d_model, d_internal)
        

    def forward(self, input_vecs):
        """
        :param input_vecs: an input tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the log probabilities of each position in the input
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """
        #print("X in Transformer Layer forward")
        self_attn, scores = self.attention_layer(input_vecs, input_vecs, input_vecs)
        input_vecs = torch.add(input_vecs, self_attn)
        ff = self.feed_forward(input_vecs)
        input_vecs = torch.add(input_vecs , ff)
        return input_vecs, scores

# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        #print("X in Positional Embedding init")
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        #print("X in Positional Embedding forward")
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:

    #print(train[:3])

    vocab_size = 27
    num_positions = 20
    d_model = 128
    d_internal = 32
    num_classes = 3
    num_layers = 4
    num_epochs = 5

    
    print("Model Initialization")
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.NLLLoss()

    print("Model Training")
    for i in range(num_epochs):
        for j in range(len(train)):
            probs, attn = model(train[j].input_tensor)
            Nllloss = loss(probs, train[j].output_tensor)
            model.zero_grad()
            Nllloss.backward()
            optimizer.step()
    
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
