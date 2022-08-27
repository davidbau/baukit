import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TokenizedDataset(Dataset):
    """
    Converts a dataset of text samples into a dataset of token sequences,
    as converted by a supplied tokenizer. The tokens come along with position
    ids and attention masks, they can be supplied direcly to the model.
    """
    def __init__(self, text_dataset, tokenizer=None, maxlen=None, field="text"):
        self.text_dataset = text_dataset
        self.field = field
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, i):
        text = self.text_dataset[i]
        if self.field is not None:
            text = text[self.field]
        token_list = self.tokenizer.encode(
            text, truncation=True, max_length=self.maxlen
        )
        position_ids = list(range(len(token_list)))
        attention_mask = [1] * len(token_list)
        return dict(
            input_ids=torch.tensor(token_list),
            position_ids=torch.tensor(position_ids),
            attention_mask=torch.tensor(attention_mask)
        )


def move_to(device, *containers):
    """
    Moves tensors or containers of tensors to the specified device,
    moving tensors in-place within arrays, dictionaries, and Modules.

    Example:
          [moved_a, moved_b] = move_to('cuda', a, b)

    If arguments are arrays or dictionaries or torch.nn.Modules
    containing tensors, the tensors are moved to the given device and
    replaced in-place without making a newe instance of the container.
    """
    containers = list(containers)
    for j, container in enumerate(containers):
        if isinstance(container, torch.nn.Module):
            container.to(device)
        elif isinstance(container, (list, dict)):
            if isinstance(container, dict):
                g = list(container.items())
            else:
                g = enumerate(container)
            for i, v in g:
                [moved] = move_to(device, v)
                if moved is not v:
                    container[i] = moved
        elif isinstance(container, (torch.nn.Parameter, torch.Tensor)):
            containers[j] = container.to(device)
    return containers


def length_collation(token_size):
    """
    Sorts a batch of sequences and breaks it up into subbatches
    of same-sized sequences, padding as needed.  Each batch
    has no more than token_size total tokens (or a single
    sequence, if the sequence happens to be larger).
    """

    def collate_fn(items):
        items = sorted(items, key=lambda x: -len(x["input_ids"]))
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = 0
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        return batches

    return collate_fn


def make_padded_batch(items):
    """
    Pads sequences in a batch, so they are all the same length as the longest.
    """
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}

    def join_items(items, k):
        nonempty_data = [d[k] for d in items if len(d["input_ids"])]
        if torch.is_tensor(items[0][k]):
            return pad_sequence(nonempty_data, batch_first=True)
        return nonempty_data

    return { k: join_items(items, k) for k, v in items[0].items() }


def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]
