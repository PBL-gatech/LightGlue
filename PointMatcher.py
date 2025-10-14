import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is optional
    plt = None

from lightglue import ALIKED, DISK, DoGHardNet, LightGlue, SIFT, SuperPoint
from lightglue.utils import load_image, match_pair

_EXTRACTOR_MAP = {
    "superpoint": SuperPoint,
    "disk": DISK,
    "aliked": ALIKED,
    "sift": SIFT,
    "doghardnet": DoGHardNet,
}

PathLike = Union[str, Path]


class PointMatcher:
    """High-level interface to measure LightGlue matching latency."""

    def __init__(
        self,
        *,
        features: str = "superpoint",
        device: Optional[str] = None,
        extractor_conf: Optional[Dict] = None,
        matcher_conf: Optional[Dict] = None,
    ) -> None:
        self.features = features.lower()
        if self.features not in _EXTRACTOR_MAP:
            supported = ", ".join(sorted(_EXTRACTOR_MAP))
            raise ValueError(f"Unsupported features '{features}'. Choose from: {supported}.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        extractor_kwargs = dict(extractor_conf or {})
        if "max_num_keypoints" not in extractor_kwargs and self.features == "superpoint":
            extractor_kwargs["max_num_keypoints"] = 2048

        extractor_cls = _EXTRACTOR_MAP[self.features]
        self.extractor = extractor_cls(**extractor_kwargs).eval().to(self.device)

        matcher_kwargs = dict(matcher_conf or {})
        self.matcher = LightGlue(features=self.features, **matcher_kwargs).eval().to(self.device)

    def match(
        self,
        image0: Union[PathLike, torch.Tensor],
        image1: Union[PathLike, torch.Tensor],
        *,
        load_conf: Optional[Dict] = None,
        **preprocess,
    ) -> Dict[str, object]:
        """Match two images and report latency.

        Args:
            image0/image1: Either file paths or pre-loaded ``torch.Tensor`` images.
            load_conf: Extra kwargs forwarded to ``lightglue.utils.load_image`` when the
                inputs are file paths. Example: ``{\"resize\": 1024}``.
            **preprocess: Additional preprocessing configuration for the extractor.

        Returns:
            Dict containing ``feats0``, ``feats1``, ``matches`` and ``latency``.
        """
        image0_tensor = self._prepare_image(image0, load_conf)
        image1_tensor = self._prepare_image(image1, load_conf)

        start = time.perf_counter()
        feats0, feats1, matches = match_pair(
            self.extractor, self.matcher, image0_tensor, image1_tensor, device=self.device, **preprocess
        )
        latency = time.perf_counter() - start

        keypoints_pair = self._get_matched_keypoints(feats0, feats1, matches)
        center_shift = self._calculate_shift(keypoints_pair, feats0, feats1)
        overlay = self._match_patch(image0_tensor, image1_tensor, keypoints_pair)

        return {
            "feats0": feats0,
            "feats1": feats1,
            "matches": matches,
            "latency": latency,
            "center_shift": center_shift,
            "overlay": overlay,
        }

    def _prepare_image(
        self, image: Union[PathLike, torch.Tensor], load_conf: Optional[Dict]
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            load_kwargs = dict(load_conf or {})
            tensor = load_image(str(image), **load_kwargs)
        return tensor.to(self.device)

    def _get_matched_keypoints(
        self,
        feats0: Dict[str, torch.Tensor],
        feats1: Dict[str, torch.Tensor],
        matches: Dict[str, torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        match_indices = matches.get("matches")
        if match_indices is None or match_indices.ndim != 2 or match_indices.shape[1] != 2:
            return None
        if match_indices.numel() == 0:
            return None

        valid = (match_indices >= 0).all(dim=1)
        if not valid.any().item():
            return None

        match_indices = match_indices[valid].to(device=feats0["keypoints"].device, dtype=torch.long)
        if match_indices.numel() == 0:
            return None

        points0 = feats0["keypoints"][match_indices[:, 0]]
        points1 = feats1["keypoints"][match_indices[:, 1]]
        if points0.shape[0] == 0 or points1.shape[0] == 0:
            return None

        return points0, points1

    def _calculate_shift(
        self,
        keypoints_pair: Optional[Tuple[torch.Tensor, torch.Tensor]],
        feats0: Dict[str, torch.Tensor],
        feats1: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, float]]:
        """Estimate translation between image centers from matched keypoints."""
        if keypoints_pair is None:
            return None

        points0, points1 = keypoints_pair
        translation = (points1 - points0).mean(dim=0)
        if not torch.isfinite(translation).all():
            return None

        size0 = feats0.get("image_size")
        size1 = feats1.get("image_size")
        if size0 is not None and size1 is not None:
            size0 = size0.to(translation)
            size1 = size1.to(translation)
            center0 = (size0 - 1.0) / 2.0
            center1 = (size1 - 1.0) / 2.0
        else:
            center0 = center1 = None

        result = {
            "center_dx": float(translation[0].item()),
            "center_dy": float(translation[1].item()),
            "num_matches": int(points0.shape[0]),
        }

        if center0 is not None and center1 is not None:
            center0_tuple = tuple(map(float, center0.tolist()))
            center1_tuple = tuple(map(float, center1.tolist()))
            center0_in_image1 = center0 + translation
            residual = center1 - center0_in_image1
            result.update(
                {
                    "center0": center0_tuple,
                    "center1": center1_tuple,
                    "residual_dx": float(residual[0].item()),
                    "residual_dy": float(residual[1].item()),
                }
            )

        return result

    def _match_patch(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        keypoints_pair: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Overlay image0 onto image1 by aligning matched keypoints via translation."""
        if keypoints_pair is None:
            return None

        points0, points1 = keypoints_pair
        translation = (points1 - points0).mean(dim=0)
        if not torch.isfinite(translation).all():
            return None

        if image0.shape[0] != image1.shape[0]:
            return None

        target_hw = image1.shape[-2], image1.shape[-1]
        aligned_image0 = self._apply_translation(image0, translation, target_hw=target_hw)
        overlay = 0.5 * aligned_image0 + 0.5 * image1.to(aligned_image0)
        return overlay.clamp(0.0, 1.0).detach().cpu()

    def _apply_translation(
        self,
        image: torch.Tensor,
        translation: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Translate `image` by `translation` (dx, dy) using bilinear sampling."""
        if image.dim() != 3:
            raise ValueError("Expected image tensor with shape (C, H, W).")

        _, src_h, src_w = image.shape
        tgt_h, tgt_w = target_hw if target_hw is not None else (src_h, src_w)

        if src_h <= 1 or src_w <= 1:
            return image.clone()

        device = image.device
        dtype = image.dtype

        # Create target pixel coordinate grid.
        ys = torch.arange(tgt_h, device=device, dtype=dtype)
        xs = torch.arange(tgt_w, device=device, dtype=dtype)
        if hasattr(torch, "meshgrid"):
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        else:
            grid_y, grid_x = torch.meshgrid(ys, xs)

        # Shift coordinates by translation to sample from source image.
        src_x = grid_x - translation[0]
        src_y = grid_y - translation[1]

        if src_w > 1:
            src_x_norm = (src_x / (src_w - 1)) * 2 - 1
        else:
            src_x_norm = src_x * 0
        if src_h > 1:
            src_y_norm = (src_y / (src_h - 1)) * 2 - 1
        else:
            src_y_norm = src_y * 0

        grid = torch.stack((src_x_norm, src_y_norm), dim=-1).unsqueeze(0)
        image_batch = image.unsqueeze(0)

        warped = F.grid_sample(
            image_batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped.squeeze(0)

if __name__ == "__main__":
    IMAGE0_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\cell_4.webp")
    IMAGE1_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\36712_1760473097.769556.webp")

    matcher = PointMatcher(features="superpoint")
    result = matcher.match(IMAGE0_PATH, IMAGE1_PATH)

    latency_ms = result["latency"] * 1e3
    print(f"Latency: {latency_ms:.2f} ms")
    shift = result.get("center_shift")
    if shift is None:
        print("Center shift: unavailable (no valid matches).")
    else:
        print(
            f"Center shift: dx={shift['center_dx']:.2f} px, dy={shift['center_dy']:.2f} px "
            f"(from {shift['num_matches']} matches)"
        )
        if "residual_dx" in shift and "residual_dy" in shift:
            print(
                f"Residual center error after translation: "
                f"dx={shift['residual_dx']:.2f} px, dy={shift['residual_dy']:.2f} px"
            )

    overlay = result.get("overlay")
    if overlay is None:
        print("Overlay: unavailable.")
    else:
        print(f"Overlay image shape: {tuple(overlay.shape)}")
        if plt is None:
            print("Matplotlib not installed; cannot display overlay.")
        else:
            overlay_np = overlay.permute(1, 2, 0).numpy()
            plt.figure("LightGlue Overlay")
            plt.imshow(overlay_np)
            plt.title("Overlay: image0 warped onto image1")
            plt.axis("off")
            plt.show()
