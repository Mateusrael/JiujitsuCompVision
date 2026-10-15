# Modules

All five models return unnormalized logits with shape `(B, 10)` for the shared
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

## ResNet-18 image classifier

`models.ImageClassifier(num_classes=10, pretrained=True, image_training="head", dropout=0.0)` uses
torchvision ResNet-18 and a replacement linear classifier, selected by
`--model image`. Fresh image runs default to
`ResNet18_Weights.DEFAULT`; a first run may download these weights. The backbone
is frozen by default, including BatchNorm running statistics. `--image-training`
selects `head`, `input-head`, `layer4`, or `full`. Input is `(B, 3, 224, 224)`.

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
weights and bias. Mode `input-head` also trains only `backbone.conv1`, the first
7 × 7 convolution. All other weights and all BatchNorm running statistics,
including the initial BatchNorm, stay frozen. It trains **14,538 parameters**.
Gradients pass through the frozen stages to reach the first convolution.
Mode `layer4` also trains both residual blocks in the last
stage, leaving the initial convolution and stages 1–3 frozen. Mode `full` trains
all backbone parameters. Freezing a stage disables its gradients and keeps its
BatchNorm running statistics unchanged through every `train()`/`eval()` cycle.
BatchNorm in the trainable stages continues updating during training.
For ten classes, `head` trains 5,130 parameters and `layer4` trains 8,398,858
out of 11,181,642 total (75.11%). The last stage contains most of ResNet-18's
parameters, so freezing the earlier stages still leaves substantial capacity.

The legacy `fine_tune` argument still maps `False` to `head` and `True` to `full`
when no explicit mode is supplied. An explicit mode must agree with a supplied
boolean about whether any backbone parameters are trainable.

Optional dropout acts on the 512 pooled features before this classifier. It is
disabled by default and active during training even if the backbone is frozen.
It does not change the backbone's frozen BatchNorm behavior. Every model disables
dropout during validation and test evaluation.

## DINOv3 image classifier

`dinov3.FrozenDINOv3Classifier(num_classes=10, pretrained=True, dropout=0.0, image_training="head")` is
selected by `--model image-dinov3`. It uses Transformers 4.57.1's
`DINOv3ViTModel` with the official
[`facebook/dinov3-vitb16-pretrain-lvd1689m` checkpoint](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
The implementation is installed as a package; model loading does not fetch or
execute remote Python code. The Transformers import is lazy, so importing the
other model implementations does not require it.

ViT-B/16 uses 16 × 16 image patches, 12 transformer blocks with 12 heads,
768-dimensional embeddings, 3072-dimensional block MLPs, and four register tokens.
The final normalized CLS token is the image representation (`pooler_output`).
It passes through optional dropout and `Linear(768, 10)`: **7,690 trainable
parameters** in the default `head` mode, with roughly 86 million frozen backbone parameters.

Mode `input-head` also trains `backbone.embeddings.patch_embeddings`, the existing
`Conv2d(3, 768, kernel_size=16, stride=16)` projection, including its bias.
This gives **598,282 trainable parameters**. Transformer blocks, normalization,
CLS/register/mask tokens and all other backbone weights remain frozen.
`train()` keeps the backbone in evaluation mode in both scopes, including its
positional augmentation. Only `head` uses `no_grad()` for feature extraction;
`input-head` retains the computation graph through the frozen transformer to
reach the patch projection. This requires more activation memory and computation
than head-only training. Dropout applies only before the head, defaults to zero,
and is disabled for validation/test. Other DINOv3 training scopes are unsupported.

Input is the same `(B, 3, 224, 224)` letterboxed, ImageNet-normalized full frame
used by ResNet-18. Letterboxing is an intentional project choice to preserve
both athletes; it is not the official processor's direct resize. Features are
computed each batch; there is no feature cache.

Fresh pretrained loading requires access to the gated model and a Hugging Face
login in the training environment. Bootstrap does not download weights.
`dinov3_config.py` records the exact backend version, weights ID and explicit
backbone configuration without importing Torch. `pretrained=False` constructs
that architecture locally for complete checkpoint reconstruction or synthetic
smoke tests. Frozen random features are not a useful pretrained baseline.

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
New image runs also record `image_training`; legacy head/full definitions with
only `fine_tune` are still loadable. Resume restores the selected mode and rejects
scope changes so that optimizer state always matches the trainable parameters.
DINOv3 contracts additionally record its pretrained model identity, backend
version, complete backbone configuration, frozen scope and CLS pooling. The
checkpoint contains the backbone and classifier state, so resume/evaluation
requires the installed backend but no model download or Hub authentication.

## Optional compiled execution

`execution.compile_model(model, enabled=False, mode="default")` returns the
original model when disabled or `torch.compile(model, mode=mode)` when enabled.
The whole classifier is passed to the compiler. Architectures and their blocks
have no compile decorators. `COMPILE_MODES` lists `default`, `reduce-overhead`,
`max-autotune`, and `max-autotune-no-cudagraphs`;
`validate_compile_settings(enabled, mode)` validates the execution options
without importing PyTorch.

Callers retain the original model for optimizer parameters and checkpoint state,
using the returned wrapper for forward execution. Loading restores raw weights
before optionally compiling. Compilation is separate from `architecture_config`
and can be switched on/off without changing model state keys. Initial calls can
compile lazily; the helper reports this and preserves compiler error details.
DINOv3 uses the same whole-model wrapper. Native CUDA/Inductor execution and
performance must be verified on the target DGX; validation of the earlier four
models does not establish DINOv3 compatibility or a speedup.
