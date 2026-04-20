import numpy as np

def entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)

def cross_entropy(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, eps, 1.0)
    return -np.sum(p * np.log(q))  

def KL(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    e = -np.sum(p * np.log(p), axis=-1)
    ce = -np.sum(p * np.log(q), axis=-1)
    return ce - e

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.6, 0.3, 0.1])

H_p = entropy(p)
H_pq = cross_entropy(p, q)
KL_pq = KL(p, q)

print("H(p)      =", H_p)
print("H(p, q)   =", H_pq)
print("KL(p||q)  =", KL_pq)
print("H(p)+KL   =", H_p + KL_pq)