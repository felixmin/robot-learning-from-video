Yes — what you describe is **a very common setup**. In ML it’s usually called **multi-task learning with partially labeled data**. The standard solution is **exactly what you suggested: masking the loss for samples where the label is missing**. ([dlology.com][1])

I'll address your two concerns separately.

---

# 1. Is masking the loss standard practice?

Yes. The typical implementation is:

[
L = L_1(x,y_1) + m \cdot L_2(x,y_2)
]

where

* (m = 1) if label exists
* (m = 0) if label missing

In code this is usually done by computing **per-sample loss** and multiplying by a mask before averaging.

Example pattern (PyTorch):

```python
loss1 = loss_fn1(pred1, y1)                # shape [B]
loss2 = loss_fn2(pred2, y2)                # shape [B]

loss2 = loss2 * mask                       # mask: 1 if label exists

loss = loss1.mean() + loss2.sum() / mask.sum()
```

Masking missing labels is widely used in:

* multi-task learning
* multi-label classification
* semi-supervised learning
* datasets with partial annotations

Many papers explicitly train models where **some tasks only have labels for a subset of samples**. ([Fibion][2])

So **architecturally you are doing the right thing**.

---

# 2. Is the gradient too noisy with only ~6 samples?

Your intuition is correct, but it’s **usually not a problem** for two reasons.

### Reason 1 — gradients accumulate over batches

Even if each batch has only ~6 samples:

* over **1000 batches → 6000 gradient contributions**

So the head still receives many updates.

---

### Reason 2 — shared representation stabilizes learning

Because the backbone is shared:

```
backbone(x)
   ├── head1 -> loss1 (64 samples)
   └── head2 -> loss2 (6 samples)
```

the backbone learns **stable features from loss1**, which makes the second head easier to learn.

This is actually one of the motivations for **multi-task learning**.

---

# 3. The *real* issue: loss imbalance

The real risk is **loss imbalance**, not gradient noise.

In your case:

```
loss = loss_full + loss_sparse
```

But:

* loss_full → 64 samples
* loss_sparse → 6 samples

If you average over batch naively, the sparse task can get **10× weaker gradients**.

---

# 4. The standard fix (important)

Compute the mean **only over available labels**:

[
L_2 = \frac{\sum_i m_i L_{2,i}}{\sum_i m_i}
]

instead of

[
\frac{1}{B}\sum_i m_i L_{2,i}
]

So:

```python
loss2 = (loss2 * mask).sum() / mask.sum()
```

This ensures **both tasks have similar scale**.

---

# 5. Additional techniques people use

If the sparse task is still unstable:

### (A) Task weighting

Common in multi-task learning.

[
L = L_1 + \lambda L_2
]

or dynamic weighting (e.g. **uncertainty weighting**, GradNorm).

---

### (B) Task-specific batches

Sometimes people construct batches like:

```
batch = 32 full labels
batch = 32 sparse labels
```

This guarantees enough examples per task.

---

### (C) Oversample labeled data

If only 10% of samples have label2:

```
sample(label2 data) with higher probability
```

Very common in practice.

---

### (D) Two dataloaders (very common in research)

Alternate updates:

```
step1: batch from dataset1 → loss1
step2: batch from dataset2 → loss2
```

Used in many multitask setups.

---

# 6. When masking becomes problematic

It only becomes problematic if:

* extremely sparse labels (e.g. **<1%**)
* very small batches
* unstable targets

But **10% labels is usually completely fine**.

---

# 7. Quick rule of thumb

Your setup is perfectly reasonable if:

```
(# labeled samples for task2 per epoch) > 10k
```

Even **a few thousand** can work well.

---

✅ **Conclusion**

Your approach is **standard practice**:

* shared backbone
* task-specific heads
* masked loss for missing labels

Just make sure you **normalize the sparse loss by the number of labeled samples**.

---

If you'd like, I can also show you **a very common mistake in PyTorch implementations of this setup that silently breaks training**. It's something I see even in research codebases.

[1]: https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/?utm_source=chatgpt.com "How to Multi-task learning with missing labels in Keras | DLology"
[2]: https://web.fibion.com/articles/multi-task-learning-imu-movement/?utm_source=chatgpt.com "Multi-task IMU learning for movement"
