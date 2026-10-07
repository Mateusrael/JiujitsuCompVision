# Modules

`models.PoseMLP(num_classes=10)` uses 102 -> 102 -> 34 -> 10 linear layers with
ReLU between hidden layers. It consumes `(B, 102)` and returns unnormalized logits.

`models.ImageClassifier(num_classes=10, pretrained=True, fine_tune=False)` uses
torchvision ResNet-18 and a new linear classifier. Fresh image runs default to
`ResNet18_Weights.DEFAULT`; a first run may download these weights. The backbone
is frozen by default, including BatchNorm running statistics. `--fine-tune`
enables all backbone parameters. Input is `(B, 3, 224, 224)`.

`architecture_config` describes the exact architecture/preprocessing contract.
`build_model` validates that contract. Checkpoint reconstruction always disables
pretrained downloads because all model weights are saved in the checkpoint.
