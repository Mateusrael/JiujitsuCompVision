# Modules

All four models return unnormalized logits with shape `(B, 10)` for the shared
class order. Cross entropy consumes these logits directly; the models do not
apply a final softmax. Each input describes one frame.

## Pose MLPs

`models.PoseMLP(num_classes=10, dropout=0.0)`, selected by `--model pose`, contains three
linear layers: `102 → 102 → 34 → 10`. ReLU follows each of the two hidden layers.
There is no activation after the output layer.

`models.PoseWideMLP(num_classes=10, dropout=0.0)`, selected by `--model pose-wide`, contains
four linear layers: `102 → 512 → 128 → 32 → 10`. ReLU follows each of the three
hidden layers. Both MLPs train every parameter from random initialization and
have no normalization layers. Optional dropout follows every hidden ReLU and
defaults to zero. It is not applied to output logits.

## Joint self-attention

`pose_attention.PoseAttentionClassifier`, selected by `--model pose-attention`,
accepts the same `(B, 102)` features as the MLPs. It reshapes them internally to
`(B, 34, 3)`: athlete 1's 17 COCO joints, then athlete 2's 17 joints, each storing
normalized x, y and confidence. Tokens represent joints in one frame, not frames
in a video sequence.

The default architecture is:

```text
34 joints × 3 features
  → shared Linear(3, 128) + 34 learned joint/athlete slot embeddings
  → 4 residual blocks:
      x = x + MultiheadAttention(LayerNorm(x)), 4 heads
      x = x + MLP(LayerNorm(x)), Linear(128, 512) → GELU → Linear(512, 128)
  → LayerNorm(128)
  → mean over observed joints
  → Linear(128, 10)
```

The LayerNorm operations precede each block's attention and MLP. Each attention
head has width 32 with the defaults. Learned slot embeddings distinguish both
joint identity and athlete slot. Self-attention can relate joints within and
between athletes. All embeddings, attention projections, MLPs, normalization
parameters and classifier parameters are trained from random initialization.

Missing joints have confidence <= 0. Their input is zeroed before projection,
and they are excluded as attention keys and from mean pooling. A sample with no
observed joints uses a safe attention placeholder, then pools to zero; its mean
variant's output equals the classifier bias. With `--attention-pooling cls`,
a learned classification token is prepended and its final representation is
classified instead. This makes 35 tokens and also keeps all-missing inputs valid.

CLI architecture options are `--attention-dim` (128), `--attention-heads` (4),
`--attention-layers` (4), `--attention-mlp-dim` (512), and `--attention-pooling`
(`mean` or `cls`, default `mean`). Width must be divisible by head count and
counts must be positive.

Each block has two independent dropout rates, both zero by default:

- `--attention-dropout`: attention weights and attention output before its
  residual addition.
- `--attention-mlp-dropout`: hidden features after GELU and MLP output before
  its residual addition.

`--dropout` is a shared fallback for omitted branch flags. Explicit branch values
override it, including zero. All rates must be finite and in [0, 1). Direct
`PoseAttentionClassifier` construction uses the keyword arguments
`attention_dropout` and `mlp_dropout`, with `dropout` as the same shared fallback.

All three pose models use the same normalized paired-pose features and split.
They support the same training augmentations: athlete swapping with probability
0.5 by default, and optional horizontal mirroring with probability zero by
default. Changing architecture does not filter the dataset or require images.

## Image classifier

`models.ImageClassifier(num_classes=10, pretrained=True, fine_tune=False, dropout=0.0)` uses
torchvision ResNet-18 and a replacement linear classifier, selected by
`--model image`. Fresh image runs default to
`ResNet18_Weights.DEFAULT`; a first run may download these weights. The backbone
is frozen by default, including BatchNorm running statistics. `--fine-tune`
enables all backbone parameters. Input is `(B, 3, 224, 224)`.

For a 224 × 224 image, the backbone has a 7 × 7 stride-2 convolution, BatchNorm,
ReLU and a 3 × 3 stride-2 max pool, then four stages with channel widths
64, 128, 256 and 512. Each stage contains two residual basic blocks, each with
two 3 × 3 convolutions and BatchNorm, with ReLU between them and after the
residual addition. The first block in stages 2–4 downsamples and uses a
1 × 1 projection shortcut. Global average pooling produces 512 features.

The original ImageNet `512 → 1000` classification layer is replaced with an
identity operation inside `backbone`; `classifier = Linear(512, 10)` takes its
place in the full model. It is not stacked after the original 1,000 predictions.
The new classifier is initialized randomly. Default training updates only its
weights and bias; `--fine-tune` also updates all backbone parameters.

Optional dropout acts on the 512 pooled features before this classifier. It is
disabled by default and active during training even if the backbone is frozen.
It does not change the backbone's frozen BatchNorm behavior. Every model disables
dropout during validation and test evaluation.

## Checkpoint contract

`architecture_config` describes the exact architecture/preprocessing contract.
`build_model` validates that contract. Checkpoint reconstruction always disables
pretrained downloads because all model weights are saved in the checkpoint.
Attention checkpoints include embedding/head/block/MLP dimensions,
`attention_dropout` and `mlp_dropout` as separate effective rates,
pooling, learned slot order, masking, activation, normalization and preprocessing.
Resume preserves these choices and evaluation rebuilds the saved architecture.
MLP and image model contracts record their single `dropout` rate. All contracts
must match the complete configuration produced by `architecture_config`.
