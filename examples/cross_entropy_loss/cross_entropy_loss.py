import torch
from scripts.kernels.cross_entropy_loss import fast_cross_entropy_loss

def calculate_cross_entropy_loss():
    """
    Calculate the cross-entropy loss using randomly generated logits and labels.

    This function generates random logits and labels, computes the cross-entropy loss
    using the `fast_cross_entropy_loss` function, and prints the result.
    """
    batch_size = 32
    seq_len = 10
    vocab_size = 1000
    logits = torch.randn(batch_size, seq_len, vocab_size).cuda()
    labels = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    loss = fast_cross_entropy_loss(logits, labels)
    print("Cross-entropy loss:", loss.item())

if __name__ == "__main__":
    calculate_cross_entropy_loss()