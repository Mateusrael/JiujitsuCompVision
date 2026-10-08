"""Optional execution settings, separate from a model's saved architecture."""


COMPILE_MODES = ("default", "reduce-overhead", "max-autotune",
                 "max-autotune-no-cudagraphs")


def validate_compile_settings(enabled, mode):
    if not isinstance(enabled, bool):
        raise ValueError("compile must be a boolean")
    if mode not in COMPILE_MODES:
        raise ValueError(f"compile-mode must be one of {COMPILE_MODES}")


def compile_model(model, *, enabled=False, mode="default"):
    """Compile the whole model when requested, keeping its raw owner available.

    Callers retain ``model`` for the optimizer and checkpoint state, and use the
    returned object for forward execution. Compilation may happen lazily on the
    first batch or when a new input shape or train/eval graph is encountered.
    """
    validate_compile_settings(enabled, mode)
    if not enabled:
        return model
    import torch
    compiler = getattr(torch, "compile", None)
    if not callable(compiler):
        raise RuntimeError("This PyTorch installation does not provide torch.compile; "
                           "use --no-compile to run without compilation")
    print(f"torch.compile enabled (mode={mode}). Initial batches may be slower "
          "while model graphs compile.", flush=True)
    try:
        return compiler(model, mode=mode)
    except Exception as error:
        raise RuntimeError("torch.compile could not initialize. Check the underlying compiler "
                           "error, or use --no-compile to run without compilation") from error
