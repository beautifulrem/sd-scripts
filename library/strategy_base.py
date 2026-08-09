# base class for platform strategies. this file defines the interface for strategies

import os
import re
from typing import Any, List, Optional, Tuple, Union, Callable

import numpy as np
import torch


from library import caching
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class TokenizeStrategy:
    _strategy = None  # strategy instance: actual strategy class

    _re_attention = re.compile(
        r"""\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
        re.X,
    )

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TokenizeStrategy"]:
        return cls._strategy

    def _load_tokenizer(
        self, model_class: Any, model_id: str, subfolder: Optional[str] = None, tokenizer_cache_dir: Optional[str] = None
    ) -> Any:
        tokenizer = None
        if tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(tokenizer_cache_dir, model_id.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = model_class.from_pretrained(local_tokenizer_path)  # same for v1 and v2

        if tokenizer is None:
            tokenizer = model_class.from_pretrained(model_id, subfolder=subfolder)

        if tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        return tokenizer

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        raise NotImplementedError

class TextEncodingStrategy:
    _strategy = None  # strategy instance: actual strategy class

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncodingStrategy"]:
        return cls._strategy

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Encode tokens into embeddings and outputs.
        :param tokens: list of token tensors for each TextModel
        :return: list of output embeddings for each architecture
        """
        raise NotImplementedError


class TextEncoderOutputsCachingStrategy:
    _strategy = None  # strategy instance: actual strategy class

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: Optional[int],
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        is_weighted: bool = False,
    ) -> None:
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check
        self._is_partial = is_partial
        self._is_weighted = is_weighted

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncoderOutputsCachingStrategy"]:
        return cls._strategy

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def is_partial(self):
        return self._is_partial

    @property
    def is_weighted(self):
        return self._is_weighted

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        raise NotImplementedError

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        raise NotImplementedError

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        raise NotImplementedError

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, batch: List
    ):
        raise NotImplementedError


class LatentsCachingStrategy:
    # TODO commonize utility functions to this class, such as npz handling etc.

    _strategy = None  # strategy instance: actual strategy class

    _warned_fallback_to_old_npz = False  # to avoid spamming logs about fallback

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check

    @classmethod
    def set_strategy(cls, strategy):
        if cls._strategy is not None:
            raise RuntimeError(f"Internal error. {cls.__name__} strategy is already set")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["LatentsCachingStrategy"]:
        return cls._strategy

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def cache_suffix(self):
        raise NotImplementedError

    def get_image_size_from_disk_cache_path(self, absolute_path: str, npz_path: str) -> Tuple[Optional[int], Optional[int]]:
        w, h = os.path.splitext(npz_path)[0].split("_")[-2].split("x")
        return int(w), int(h)

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        raise NotImplementedError

    def is_disk_cached_latents_expected(
        self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool
    ) -> bool:
        raise NotImplementedError

    def cache_batch_latents(self, model: Any, batch: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        raise NotImplementedError

    def _default_is_disk_cached_latents_expected(
        self,
        latents_stride: int,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        apply_alpha_mask: bool,
        multi_resolution: bool = False,
    ) -> bool:
        """
        Args:
            latents_stride: stride of latents
            bucket_reso: resolution of the bucket
            npz_path: path to the npz file
            flip_aug: whether to flip images
            apply_alpha_mask: whether to apply alpha mask
            multi_resolution: whether to use multi-resolution latents

        Returns:
            bool
        """
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        expected_latents_size = (bucket_reso[1] // latents_stride, bucket_reso[0] // latents_stride)  # bucket_reso is (W, H)

        # e.g. "_32x64", HxW
        key_reso_suffix = f"_{expected_latents_size[0]}x{expected_latents_size[1]}" if multi_resolution else ""

        try:
            npz = np.load(npz_path)

            # Legacy caches may use an unsuffixed ``latents`` key; current Anima
            # multi-resolution caches use a resolution suffix.
            if "latents" + key_reso_suffix not in npz and "latents" not in npz:
                return False
            if flip_aug and ("latents_flipped" + key_reso_suffix not in npz and "latents_flipped" not in npz):
                return False
            if apply_alpha_mask and ("alpha_mask" + key_reso_suffix not in npz and "alpha_mask" not in npz):
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    # TODO remove circular dependency for ImageInfo
    def _default_cache_batch_latents(
        self,
        encode_by_vae: Callable,
        vae_device: torch.device,
        vae_dtype: torch.dtype,
        image_infos: List,
        flip_aug: bool,
        apply_alpha_mask: bool,
        random_crop: bool,
        multi_resolution: bool = False,
    ):
        """
        Default implementation for cache_batch_latents. Image loading, VAE, flipping, alpha mask handling are common.

        Args:
            encode_by_vae: function to encode images by VAE
            vae_device: device to use for VAE
            vae_dtype: dtype to use for VAE
            image_infos: list of ImageInfo
            flip_aug: whether to flip images
            apply_alpha_mask: whether to apply alpha mask
            random_crop: whether to random crop images
            multi_resolution: whether to use multi-resolution latents

        Returns:
            None
        """
        img_tensor, alpha_masks, original_sizes, crop_ltrbs = caching.load_images_and_masks_for_caching(
            image_infos, apply_alpha_mask, random_crop
        )
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latents_tensors = encode_by_vae(img_tensor).to("cpu")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = encode_by_vae(img_tensor).to("cpu")
        else:
            flipped_latents = [None] * len(latents_tensors)

        # for info, latents, flipped_latent, alpha_mask in zip(image_infos, latents_tensors, flipped_latents, alpha_masks):
        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_tensors[i]
            flipped_latent = flipped_latents[i]
            alpha_mask = alpha_masks[i]
            original_size = original_sizes[i]
            crop_ltrb = crop_ltrbs[i]

            latents_size = latents.shape[-2:]  # H, W (supports both 4D and 5D latents)
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}" if multi_resolution else ""  # e.g. "_32x64", HxW

            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_npz, latents, original_size, crop_ltrb, flipped_latent, alpha_mask, key_reso_suffix
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                info.alpha_mask = alpha_mask

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        For single resolution architectures (currently no architecture is single resolution specific). Kept for reference.

        Args:
            npz_path (str): Path to the npz file.
            bucket_reso (Tuple[int, int]): The resolution of the bucket.

        Returns:
            Tuple[
                Optional[np.ndarray],
                Optional[List[int]],
                Optional[List[int]],
                Optional[np.ndarray],
                Optional[np.ndarray]
            ]: Latent np tensors, original size, crop (left top, right bottom), flipped latents, alpha mask
        """
        return self._default_load_latents_from_disk(None, npz_path, bucket_reso)

    def _default_load_latents_from_disk(
        self, latents_stride: Optional[int], npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Args:
            latents_stride (Optional[int]): Stride for latents. If None, load all latents.
            npz_path (str): Path to the npz file.
            bucket_reso (Tuple[int, int]): The resolution of the bucket.

        Returns:
            Tuple[
                Optional[np.ndarray],
                Optional[List[int]],
                Optional[List[int]],
                Optional[np.ndarray],
                Optional[np.ndarray]
            ]: Latent np tensors, original size, crop (left top, right bottom), flipped latents, alpha mask
        """
        if latents_stride is None:
            key_reso_suffix = ""
        else:
            expected_latents_size = (bucket_reso[1] // latents_stride, bucket_reso[0] // latents_stride)  # bucket_reso is (W, H)
            key_reso_suffix = f"_{expected_latents_size[0]}x{expected_latents_size[1]}"  # e.g. "_32x64", HxW

        npz = np.load(npz_path)
        if "latents" + key_reso_suffix not in npz:
            # raise ValueError(f"latents{key_reso_suffix} not found in {npz_path}")
            # Fallback to old npz without resolution suffix
            if "latents" not in npz:
                raise ValueError(f"latents not found in {npz_path} (either with or without resolution suffix: {key_reso_suffix})")
            if not self._warned_fallback_to_old_npz:
                logger.warning(
                    f"latents{key_reso_suffix} not found in {npz_path}. Falling back to latents without resolution suffix (old npz). This warning will only be shown once. To avoid this warning, please re-cache the latents with the latest version."
                )
                self._warned_fallback_to_old_npz = True
            key_reso_suffix = ""

        latents = npz["latents" + key_reso_suffix]
        original_size = npz["original_size" + key_reso_suffix].tolist()
        crop_ltrb = npz["crop_ltrb" + key_reso_suffix].tolist()
        flipped_latents = npz["latents_flipped" + key_reso_suffix] if "latents_flipped" + key_reso_suffix in npz else None
        alpha_mask = npz["alpha_mask" + key_reso_suffix] if "alpha_mask" + key_reso_suffix in npz else None
        return latents, original_size, crop_ltrb, flipped_latents, alpha_mask

    def save_latents_to_disk(
        self,
        npz_path,
        latents_tensor,
        original_size,
        crop_ltrb,
        flipped_latents_tensor=None,
        alpha_mask=None,
        key_reso_suffix="",
    ):
        """
        Args:
            npz_path (str): Path to the npz file.
            latents_tensor (torch.Tensor): Latent tensor
            original_size (List[int]): Original size of the image
            crop_ltrb (List[int]): Crop left top right bottom
            flipped_latents_tensor (Optional[torch.Tensor]): Flipped latent tensor
            alpha_mask (Optional[torch.Tensor]): Alpha mask
            key_reso_suffix (str): Key resolution suffix

        Returns:
            None
        """
        kwargs = {}

        if os.path.exists(npz_path):
            # load existing npz and update it
            npz = np.load(npz_path)
            for key in npz.files:
                kwargs[key] = npz[key]

        # TODO float() is needed if vae is in bfloat16. Remove it if vae is float16.
        kwargs["latents" + key_reso_suffix] = latents_tensor.float().cpu().numpy()
        kwargs["original_size" + key_reso_suffix] = np.array(original_size)
        kwargs["crop_ltrb" + key_reso_suffix] = np.array(crop_ltrb)
        if flipped_latents_tensor is not None:
            kwargs["latents_flipped" + key_reso_suffix] = flipped_latents_tensor.float().cpu().numpy()
        if alpha_mask is not None:
            kwargs["alpha_mask" + key_reso_suffix] = alpha_mask.float().cpu().numpy()
        np.savez(npz_path, **kwargs)
