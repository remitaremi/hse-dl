import torch
from datasets import load_dataset

torch.manual_seed(1)

training_data = load_dataset("conll2000", split='train')
training_data = training_data[:3000]
training_data

word_to_ix = {}
for tokens in training_data['tokens']:
    for word in tokens:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)

word_to_ix

EMBEDDING_DIM = 500
HIDDEN_DIM = 500

class LSTMTagger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(10033, EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)

        self.pos_predictor = torch.nn.Linear(HIDDEN_DIM, 44)

    def forward(self, token_ids):
        embeds = self.embedding_layer(token_ids)
        lstm_out, _ = self.lstm(embeds.view(len(token_ids), 1, -1))
        logits = self.pos_predictor(lstm_out.view(len(token_ids), -1))
        probs = torch.nn.functional.softmax(logits, dim=1)

        return probs


model = LSTMTagger()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


for epoch in range(20):  # again, normally you would NOT do 300 epochs, it is toy data
    for i in range(len(training_data)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(training_data['tokens'][i], word_to_ix)
        targets = torch.tensor(training_data['pos_tags'][i], dtype=torch.long)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data['tokens'][0], word_to_ix)
    tag_scores = model(inputs)

    print(tag_scores)