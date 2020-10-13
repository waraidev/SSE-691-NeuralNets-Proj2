import math

from torchtext.data import Iterator, batch

global max_src_in_batch, max_trg_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."

    global max_src_in_batch, max_trg_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_trg_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_trg_in_batch = max(max_trg_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    trg_elements = count * max_trg_in_batch
    return max(src_elements, trg_elements)


class FastIterator(Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in batch(d, self.batch_size * 100):
                    p_batch = batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in batch(self.data(), self.batch_size,
                           self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
