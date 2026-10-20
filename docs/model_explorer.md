# Model explorer notebook

The notebook at `notebooks/model_explorer.ipynb` runs in the
project checkout with access to `Data/` and `trainings/`. It reads local logs
and loads saved checkpoints for inference. It does not train models.

## JupyterLab setup

From the project directory, install the optional notebook packages in the
same virtual environment used for training, then register its kernel:

```bash
.venv/bin/python -m pip install -r install/requirements-notebook.txt
.venv/bin/python -m ipykernel install --user \
  --name jiujitsu-compvision --display-name "JiujitsuCompVision (.venv)"
```

Open the notebook in JupyterLab and select **JiujitsuCompVision (.venv)**.
The setup cell prints the Python executable and PyTorch version so the active
environment is visible. Registering the kernel is a one-time step per
environment; activating a terminal's venv does not change a notebook's kernel.
Refresh JupyterLab if the new kernel does not appear immediately.

## Explore runs

Run the setup cells and the metrics-browser cell. Select a model family and
run, then choose training curves, per-class recall, a confusion matrix, or the
saved configuration. **Refresh runs** reads newly logged epochs without
restarting the kernel. Catalog status describes the log, not whether a process
is still running.

Pose curves show the first 20 logged epochs separately by default. Change
**Pose initial** to adjust the boundary; `0` shows the full trajectory.
Class figures default to the checkpoint selected by pooled validation macro
F1. Enter `last` or a logged epoch number to inspect another epoch.

The notebook also includes a code cell for comparing selected runs. Comparisons
require matching annotation and split hashes; image and pose models can still
have different eligible populations.

## Inspect predictions

Set the annotation, split, and image paths in the configuration cell. The
notebook validates the manifest and checks its hashes against the checkpoint.
If a historical run used another split, choose that split explicitly.

Select a run in the metrics browser, then click **Load selected run** in the
prediction browser. Choose `best` or `last` and a device such as `cpu` or
`cuda:4`. CUDA indexes refer to the devices visible to the kernel process.
Only one model stays loaded at a time; **Unload model** releases it.

Filter by partition, class, source, evaluation type, or frame range. Validation
is the default exploration partition. Test remains an explicit choice; repeated
inspection of test examples should not guide model selection.

- **New random samples** draws a fresh batch; **Repeatable sample** uses the
  entered seed.
- **Incorrect**, **Correct**, and **Lowest confidence** search a bounded random
  pool, then display at most the requested number. This is not a full-dataset
  evaluation.
- **Predict this image** looks up an exact image ID within the current filters.
- **Show model input** shows the actual preprocessed tensor for a displayed
  sample: normalized joint features or the letterboxed image.
- Pose overlay confidence changes only the visualization, never model inputs.

Pose inference retains the project's two-athlete and paired-view eligibility
rules in every partition. Augmentation and dropout are disabled for inference.
Full project checkpoints include the image backbones, so loading them does not
download pretrained weights or require a new Hugging Face login.
Softmax scores are displayed as model scores, not calibrated probabilities.
Use this notebook with checkpoints produced by this project that you trust.

The reusable interfaces live in `src.Eval.exploration`,
`src.Eval.exploration_plots`, and `src.Eval.exploration_ui`; they can also be
called directly from code cells. Clear notebook outputs before committing
executed copies.

If this JupyterLab server cannot display widgets, the same operations work
directly in code cells:

```python
run = load_run(TRAININGS_DIR, "your-run-directory")
display(plot_training_curves(run, pose_initial_epochs=20))
model = ModelExplorer.from_run(TRAININGS_DIR / run.name, data, device=DEVICE)
try:
    rows = data.sample(model.model_name, split="val", count=3, seed=42)
    display(plot_predictions(data, rows, model.predict(rows)))
finally:
    model.close()
```
