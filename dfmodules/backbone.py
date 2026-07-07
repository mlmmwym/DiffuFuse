from __future__ import absolute_import, division, print_function

import os
import sys

import requests
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging


BACKBONE_MODEL_NAME = "microsoft/deberta-v3-large"
BACKBONE_CACHE_NAME = "difffuse_backbone"
hf_logging.set_verbosity_error()


def progress_bar(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    return tqdm(*args, **kwargs)


def configure_huggingface_endpoint(args):
    if args.hf_endpoint:
        endpoint = args.hf_endpoint.rstrip("/")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("--hf_endpoint must start with http:// or https://, got {}".format(args.hf_endpoint))
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_CO_RESOLVE_ENDPOINT"] = endpoint
        print("Using Hugging Face endpoint: {}".format(endpoint))
        return

    for env_name in ("HF_ENDPOINT", "HUGGINGFACE_CO_RESOLVE_ENDPOINT"):
        endpoint = os.environ.get(env_name)
        if endpoint and not endpoint.startswith(("http://", "https://")):
            raise ValueError("{} must start with http:// or https://, got {}".format(env_name, endpoint))


def request_with_relative_redirects(url, stream=True, timeout=30):
    current_url = url
    for _ in range(8):
        response = requests.get(current_url, stream=stream, timeout=timeout, allow_redirects=False)
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise RuntimeError("Redirect without Location header.")
            current_url = requests.compat.urljoin(current_url, location)
            continue
        return response
    raise RuntimeError("Too many redirects while downloading backbone.")


def download_file(url, output_path):
    tmp_path = output_path + ".tmp"
    response = request_with_relative_redirects(url, stream=True)
    if response.status_code == 404:
        response.close()
        return False
    if response.status_code >= 400:
        status_code = response.status_code
        response.close()
        raise RuntimeError("Backbone download failed with HTTP status {}.".format(status_code))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_size = int(response.headers.get("Content-Length", 0))
    with open(tmp_path, "wb") as handle:
        progress = progress_bar(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="downloading backbone",
        )
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
        finally:
            progress.close()
    response.close()
    os.replace(tmp_path, output_path)
    return True


def prepare_backbone_files(args):
    if not args.manual_mirror_download:
        return BACKBONE_MODEL_NAME

    endpoint = args.hf_endpoint.rstrip("/") if args.hf_endpoint else "https://hf-mirror.com"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_root = args.cache_dir or os.path.join(project_root, "hf_models")
    local_dir = os.path.join(cache_root, BACKBONE_CACHE_NAME)
    required_files = ["config.json", "pytorch_model.bin", "spm.model"]
    optional_files = ["tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]

    missing_required = [
        file_name
        for file_name in required_files
        if not (os.path.exists(os.path.join(local_dir, file_name)) and os.path.getsize(os.path.join(local_dir, file_name)) > 0)
    ]
    missing_optional = [
        file_name
        for file_name in optional_files
        if not (os.path.exists(os.path.join(local_dir, file_name)) and os.path.getsize(os.path.join(local_dir, file_name)) > 0)
    ]
    if missing_required or missing_optional:
        print("downloading backbone")

    for file_name in required_files + optional_files:
        output_path = os.path.join(local_dir, file_name)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            continue
        url = "{}/{}/resolve/main/{}".format(endpoint, BACKBONE_MODEL_NAME, file_name)
        ok = download_file(url, output_path)
        if not ok and file_name in required_files:
            raise RuntimeError("Required backbone file not found.")
    return local_dir


def load_backbone(backbone_path=None, cache_dir=None, local_files_only=False):
    return AutoModel.from_pretrained(
        backbone_path or BACKBONE_MODEL_NAME,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def load_backbone_tokenizer(backbone_path=None, cache_dir=None, local_files_only=False):
    return AutoTokenizer.from_pretrained(
        backbone_path or BACKBONE_MODEL_NAME,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def patch_backbone_xsoftmax_for_torch():
    try:
        import transformers.models.deberta_v2.modeling_deberta_v2 as backbone_modeling
    except Exception:
        return
    if not hasattr(backbone_modeling, "XSoftmax"):
        return
    if not hasattr(backbone_modeling, "_softmax_backward_data"):
        return

    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        output = saved[-1]
        input_grad = backbone_modeling._softmax_backward_data(
            grad_output,
            output,
            ctx.dim,
            output.dtype,
        )
        return input_grad, None, None

    backbone_modeling.XSoftmax.backward = staticmethod(backward)
