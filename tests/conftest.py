from aegis.runtime import configure_torch_threads

# PyTorch's default thread count is ~25x slower on hybrid laptop CPUs (see aegis/runtime.py)
configure_torch_threads()
