import numpy as np

def softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def one_hot(y, num_classes):
    y = np.asarray(y, dtype=np.int64)
    out = np.zeros((y.shape[0], num_classes), dtype=np.int64)
    out[np.arange(y.shape[0]), y] = 1.0

    



logits = np.array([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3],
    [1.2, 0.7, 3.1]
])

y = np.array([0, 1, 2])