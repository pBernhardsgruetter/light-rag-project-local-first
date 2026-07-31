"""Extractor module package initialization."""
import sys
import os

if not hasattr(sys, "get_int_max_str_digits"):
    def get_int_max_str_digits() -> int:
        return 4300
    sys.get_int_max_str_digits = get_int_max_str_digits

if not hasattr(sys, "set_int_max_str_digits"):
    def set_int_max_str_digits(maxdigits: int) -> None:
        pass
    sys.set_int_max_str_digits = set_int_max_str_digits

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

__version__ = "0.1.0"

try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as modeling_utils
    modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass
