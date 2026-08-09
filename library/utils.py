import logging
import random
import sys
import threading
from typing import *

import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np


def fire_in_thread(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()


# region Logging


def add_logging_arguments(parser):
    parser.add_argument(
        "--console_log_level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level, default is INFO / ログレベルを設定する。デフォルトはINFO",
    )
    parser.add_argument(
        "--console_log_file",
        type=str,
        default=None,
        help="Log to a file instead of stderr / 標準エラー出力ではなくファイルにログを出力する",
    )
    parser.add_argument("--console_log_simple", action="store_true", help="Simple log output / シンプルなログ出力")


def setup_logging(args=None, log_level=None, reset=False):
    if logging.root.handlers:
        if reset:
            # remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    # log_level can be set by the caller or by the args, the caller has priority. If not set, use INFO
    if log_level is None and args is not None:
        log_level = args.console_log_level
    if log_level is None:
        log_level = "INFO"
    log_level = getattr(logging, log_level)

    msg_init = None
    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        handler = None
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler
                from rich.console import Console

                handler = RichHandler(console=Console(stderr=True))
            except ImportError:
                # print("rich is not installed, using basic logging")
                msg_init = "rich is not installed, using basic logging"

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)  # same as print
            handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)


setup_logging()
logger = logging.getLogger(__name__)

# endregion

# region PyTorch utils


def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Convert a string to a torch.dtype

    Args:
        s: string representation of the dtype
        default_dtype: default dtype to return if s is None

    Returns:
        torch.dtype: the corresponding torch.dtype

    Raises:
        ValueError: if the dtype is not supported

    Examples:
        >>> str_to_dtype("float32")
        torch.float32
        >>> str_to_dtype("fp32")
        torch.float32
        >>> str_to_dtype("float16")
        torch.float16
        >>> str_to_dtype("fp16")
        torch.float16
        >>> str_to_dtype("bfloat16")
        torch.bfloat16
        >>> str_to_dtype("bf16")
        torch.bfloat16
        >>> str_to_dtype("fp8")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fn")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fnuz")
        torch.float8_e4m3fnuz
        >>> str_to_dtype("fp8_e5m2")
        torch.float8_e5m2
        >>> str_to_dtype("fp8_e5m2fnuz")
        torch.float8_e5m2fnuz
    """
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


# endregion

# region Image utils

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def load_image(image_path, alpha=False):
    try:
        with Image.open(image_path) as image:
            if alpha:
                if not image.mode == "RGBA":
                    image = image.convert("RGBA")
            else:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
            img = np.array(image, np.uint8)
            return img
    except (IOError, OSError) as e:
        logger.error(f"Error loading file: {image_path}")
        raise e


def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
    # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
    # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

    bucket_ar = bucket_reso[0] / bucket_reso[1]
    image_ar = image_size[0] / image_size[1]
    if bucket_ar > image_ar:
        # bucketのほうが横長→縦を合わせる
        resized_width = bucket_reso[1] * image_ar
        resized_height = bucket_reso[1]
    else:
        resized_width = bucket_reso[0]
        resized_height = bucket_reso[0] / image_ar
    crop_left = (bucket_reso[0] - resized_width) // 2
    crop_top = (bucket_reso[1] - resized_height) // 2
    crop_right = crop_left + resized_width
    crop_bottom = crop_top + resized_height
    return crop_left, crop_top, crop_right, crop_bottom


def trim_and_resize_if_required(
    random_crop: bool, image: np.ndarray, reso, resized_size: Tuple[int, int], resize_interpolation: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random cropの場合のcropされた値をどうcrop left/topに反映するべきか全くアイデアがない
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


def pil_resize(image, size, interpolation):
    has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False

    if has_alpha:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    resized_pil = pil_image.resize(size, resample=interpolation)

    # Convert back to cv2 format
    if has_alpha:
        resized_cv2 = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGBA2BGRA)
    else:
        resized_cv2 = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

    return resized_cv2


def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
    resized_width: int,
    resized_height: int,
    resize_interpolation: Optional[str] = None,
):
    """
    Resize image with resize interpolation. Default interpolation to AREA if image is smaller, else LANCZOS.

    Args:
        image: numpy.ndarray
        width: int Original image width
        height: int Original image height
        resized_width: int Resized image width
        resized_height: int Resized image height
        resize_interpolation: Optional[str] Resize interpolation method "lanczos", "area", "bilinear", "bicubic", "nearest", "box"

    Returns:
        image
    """

    # Ensure all size parameters are actual integers
    width = int(width)
    height = int(height)
    resized_width = int(resized_width)
    resized_height = int(resized_height)

    if resize_interpolation is None:
        if width >= resized_width and height >= resized_height:
            resize_interpolation = "area"
        else:
            resize_interpolation = "lanczos"

    # we use PIL for lanczos (for backward compatibility) and box, cv2 for others
    use_pil = resize_interpolation in ["lanczos", "lanczos4", "box"]

    resized_size = (resized_width, resized_height)
    if use_pil:
        interpolation = get_pil_interpolation(resize_interpolation)
        image = pil_resize(image, resized_size, interpolation=interpolation)
        logger.debug(f"resize image using {resize_interpolation} (PIL)")
    else:
        interpolation = get_cv2_interpolation(resize_interpolation)
        image = cv2.resize(image, resized_size, interpolation=interpolation)
        logger.debug(f"resize image using {resize_interpolation} (cv2)")

    return image


def get_cv2_interpolation(interpolation: Optional[str]) -> Optional[int]:
    """
    Convert interpolation value to cv2 interpolation integer

    https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    """
    if interpolation is None:
        return None

    if interpolation == "lanczos" or interpolation == "lanczos4":
        # Lanczos interpolation over 8x8 neighborhood
        return cv2.INTER_LANCZOS4
    elif interpolation == "nearest":
        # Bit exact nearest neighbor interpolation. This will produce same results as the nearest neighbor method in PIL, scikit-image or Matlab.
        return cv2.INTER_NEAREST_EXACT
    elif interpolation == "bilinear" or interpolation == "linear":
        # bilinear interpolation
        return cv2.INTER_LINEAR
    elif interpolation == "bicubic" or interpolation == "cubic":
        # bicubic interpolation
        return cv2.INTER_CUBIC
    elif interpolation == "area":
        # resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
        return cv2.INTER_AREA
    elif interpolation == "box":
        # resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
        return cv2.INTER_AREA
    else:
        return None


def get_pil_interpolation(interpolation: Optional[str]) -> Optional[Image.Resampling]:
    """
    Convert interpolation value to PIL interpolation

    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
    """
    if interpolation is None:
        return None

    if interpolation == "lanczos":
        return Image.Resampling.LANCZOS
    elif interpolation == "nearest":
        # Pick one nearest pixel from the input image. Ignore all other input pixels.
        return Image.Resampling.NEAREST
    elif interpolation == "bilinear" or interpolation == "linear":
        # For resize calculate the output pixel value using linear interpolation on all pixels that may contribute to the output value. For other transformations linear interpolation over a 2x2 environment in the input image is used.
        return Image.Resampling.BILINEAR
    elif interpolation == "bicubic" or interpolation == "cubic":
        # For resize calculate the output pixel value using cubic interpolation on all pixels that may contribute to the output value. For other transformations cubic interpolation over a 4x4 environment in the input image is used.
        return Image.Resampling.BICUBIC
    elif interpolation == "area":
        # Image.Resampling.BOX may be more appropriate if upscaling
        # Area interpolation is related to cv2.INTER_AREA
        # Produces a sharper image than Resampling.BILINEAR, doesn’t have dislocations on local level like with Resampling.BOX.
        return Image.Resampling.HAMMING
    elif interpolation == "box":
        # Each pixel of source image contributes to one pixel of the destination image with identical weights. For upscaling is equivalent of Resampling.NEAREST.
        return Image.Resampling.BOX
    else:
        return None


def validate_interpolation_fn(interpolation_str: str) -> bool:
    """
    Check if a interpolation function is supported
    """
    return interpolation_str in ["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area", "box"]


# endregion

# TODO make inf_utils.py
