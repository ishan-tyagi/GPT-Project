import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64 # how many independent sequences will we proecess in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2 # 20% of immediate calcuations are dropped
#---------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all teh unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix]) # each piece becomes a row in a 4x8 tensor
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y # each batch contains 32 independent examples to train on

@torch.no_grad() # tells pytorch we don't intend to run backwards
def estimate_loss():  # averages loss over multiple batches (less noisy)
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
  """ one head of self-attention """

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False) # creates linear layers
    self.query = nn.Linear(n_embd, head_size, bias=False) # linear projections which we apply to all of our nodes
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # called a buffer, creates tril by assigning to module

    self.dropout = nn.Dropout(dropout) # dropout in order to prevent over communication of nodes
  
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) # (B,T,C)
    q = self.query(x) # (B,T,C)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1)  * C**-0.5 # (B,T,C) & (B,C,T) -> (B,T,T) # using scaled attention because it is normalized
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T) # makes sure we only communicate with prior tokens (decoder block)
    wei = F.softmax(wei, dim=-1) # (B,T,T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,C)
    out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
    return out
  
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """

  # create heads, run in parrallel, and then concatenate the outputs over channel dimension

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) 
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) 
    out = self.proj(out) # linear projection of outcome of previous compuation (line above)
    return out # projection back into the residual path way
  
class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd), # grow inner layer
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # bring it back to normal size
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x) # projection layer, going back into residual pathway
  
class Block(nn.Module): # intersperses the communication and computation
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head # since our n_embed is 32 and n_head is 4, our head_size is 8
    self.sa = MultiHeadAttention(n_head, head_size) # i.e. 4 heads of 8-dimensional self-attention (communication)
    self.ffwd = FeedForward(n_embd) # (computation)
    self.ln1 = nn.LayerNorm(n_embd) # for normalization of rows
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # utilizing residual connections for optimization, fork off and do communication and come back + normalization
    x = x + self.ffwd(self.ln2(x)) # fork off, do some computations, and then come back + normalization
    return x

# super simple bigram model
class BigramLanguageModel(nn.Module): # statistical model that predicts the probability of a word in a sequence based on the previous word

  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # thin wrapper around a vocab_size x n_embd tensor
    self.position_embedding_table = nn.Embedding(block_size, n_embd) 
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #intersperse computation and communication multiple times
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)
  
  def forward(self, idx, targets=None): # tells us how well we can generate from the model
    B, T = idx.shape
  
    # idx and targets are both (B,T) tensor of integers     
    tok_emb = self.token_embedding_table(idx) # (B,T,C) (Batch, Time, Channel) tensor (4, 8, 65)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) (allows us to consider positioning of token)
    x = tok_emb + pos_emb # (B,T,C) # encode information with token embeddings + position embeddings, we feed into self attention head
    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x)  # (B,T,vocab_size) # decoder language model head

    # logits are essentially the score of the next character in the sequence

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape          # reshaping logits so they fit what pytorch wants in cross entropy
      logits = logits.view(B*T, C)  # stretching out array so it is two dimensional
      targets = targets.view(B*T)   # making this one dimensional
      loss = F.cross_entropy(logits, targets) # measures the quality of the logits with the targets 
                                              # how well are we predicting (negative log likelyhood)
      return logits, loss                     # expected loss is -ln( 1 / (vocab size))
  
  def generate(self, idx, max_new_tokens):  #history is not really used
    #idx is (B, T) array of indices in the curretn context
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters): # trains the model

  #every once in a while evaluates the loss on train and val sets
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  # sample a batch of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) #making it start by a 1 by 1 tensor that contains 0 which is newline
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

