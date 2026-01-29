"""DMVSplat: Standalone 3DGS model with depth regularization, anchoring, and scale clamping."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from pytorch_msssim import SSIM
from torch.nn import Parameter

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """Convert c2w to gsplat world2camera matrix."""
    R = optimized_camera_to_world[:, :3, :3]
    T = optimized_camera_to_world[:, :3, 3:4]
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


def compute_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute L2 depth loss with masking.

    Args:
        pred_depth: Predicted depth [H, W, 1]
        gt_depth: Ground truth depth [H, W, 1], 0 = invalid
        mask: Optional ignore mask [H, W, 1], 1 = use, 0 = ignore

    Returns:
        Scalar loss value
    """
    valid_mask = (gt_depth > 0).float()
    if mask is not None:
        valid_mask = valid_mask * mask.float()
    diff_sq = (pred_depth - gt_depth) ** 2
    loss = (diff_sq * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    return loss


def enforce_anchor_constraint(
    means: torch.Tensor,
    anchors: torch.Tensor,
    anchor_distance: float,
) -> torch.Tensor:
    """Clamp means to be within anchor_distance of their anchors."""
    displacement = means - anchors
    distance = displacement.norm(dim=-1, keepdim=True)
    exceeded = distance > anchor_distance
    clamped = anchors + displacement / distance.clamp(min=1e-6) * anchor_distance
    return torch.where(exceeded, clamped, means)


def enforce_scale_constraint(
    scales: torch.Tensor,
    anchor_scales: torch.Tensor,
    max_scale_factor: float,
) -> torch.Tensor:
    """Clamp scales to max max_scale_factor times initial scales (per-axis)."""
    max_scales = anchor_scales + math.log(max_scale_factor)
    return torch.clamp(scales, max=max_scales)


class DMVStrategy:
    """Minimal ADC strategy - only optional opacity-based pruning."""

    def __init__(self, prune_alpha_thresh: float = 0.1, prune_every: int = 100):
        self.prune_alpha_thresh = prune_alpha_thresh
        self.prune_every = prune_every

    def step_pre_backward(self, *args, **kwargs):
        pass

    def _get_prune_mask(self, params: Dict[str, torch.nn.Parameter]) -> torch.Tensor:
        """Get mask of gaussians to keep (True = keep)."""
        opacities = torch.sigmoid(params["opacities"].squeeze(-1))
        return opacities >= self.prune_alpha_thresh

    def step_post_backward(
        self,
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        step: int,
        pruning_enable: bool = False,
    ) -> Optional[torch.Tensor]:
        """Optionally prune low-opacity gaussians. Returns prune mask if pruning occurred."""
        if not pruning_enable:
            return None

        if step > 0 and step % self.prune_every == 0:
            mask = self._get_prune_mask(params)
            if mask.sum() < len(mask):
                self._prune(params, optimizers, mask)
                return mask

        return None

    def _prune(
        self,
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        mask: torch.Tensor,
    ):
        """Remove gaussians where mask is False."""
        for name, param in params.items():
            params[name] = torch.nn.Parameter(param.data[mask])
            if name in optimizers:
                opt = optimizers[name]
                for group in opt.param_groups:
                    for i, p in enumerate(group["params"]):
                        if p is param:
                            group["params"][i] = params[name]
                            if p in opt.state:
                                state = opt.state.pop(p)
                                for key, val in state.items():
                                    if isinstance(val, torch.Tensor) and val.shape[0] == len(mask):
                                        state[key] = val[mask]
                                opt.state[params[name]] = state


@dataclass
class DMVSplatModelConfig(ModelConfig):
    """DMVSplat Model Config - standalone, no SplatfactoModel dependency."""

    _target: Type = field(default_factory=lambda: DMVSplatModel)

    # Rendering
    sh_degree: int = 3
    """Maximum degree of spherical harmonics"""
    background_color: Literal["random", "black", "white"] = "random"
    """Background color for rendering"""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """Rasterization mode"""

    # Losses
    ssim_lambda: float = 0.2
    """Weight of SSIM loss"""
    depth_regularize: bool = True
    """Whether to use depth regularization loss"""
    depth_loss_weight: float = 0.8
    """Weight for depth loss"""

    # Constraints
    enable_anchoring: bool = True
    """Whether to anchor gaussians to initial positions"""
    anchor_distance: float = 0.1
    """Maximum distance (meters) gaussians can move from anchor"""
    enable_scale_clamp: bool = True
    """Whether to clamp gaussian scales"""
    max_scale_factor: float = 2.0
    """Maximum scale growth factor per axis"""
    max_initial_scale: float = 1.0
    """Maximum initial gaussian scale in meters (world space, before scene scaling)"""

    # Pruning
    pruning_enable: bool = False
    """Whether to enable opacity-based pruning"""
    prune_alpha_thresh: float = 0.1
    """Opacity threshold for pruning"""
    prune_every: int = 100
    """Prune every N steps"""

    # Telemetry
    histogram_log_every: int = 1000
    """Log histograms every N steps"""


class DMVSplatModel(Model):
    """DMVSplat: Standalone 3DGS model for LiDAR-initialized scenes."""

    config: DMVSplatModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        # Initialize gaussian parameters
        if self.seed_points is not None:
            means = torch.nn.Parameter(self.seed_points[0])
        else:
            raise ValueError("DMVSplatModel requires seed_points from LiDAR initialization")

        # Get scene scale for converting world meters to scaled units
        scene_scale = self.kwargs.get("metadata", {}).get("scene_scale", 1.0)
        if isinstance(scene_scale, torch.Tensor):
            scene_scale = scene_scale.item()

        distances, _ = k_nearest_sklearn(means.data, 3)
        avg_dist = distances.mean(dim=-1, keepdim=True)

        # Clamp initial scale to max_initial_scale meters (converted to scaled space)
        max_scale_scaled = self.config.max_initial_scale * scene_scale
        avg_dist = torch.clamp(avg_dist, max=max_scale_scaled)

        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if self.seed_points[1].shape[0] > 0:
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
            else:
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        # Store anchors for constraints (reuse scene_scale from above)
        if self.config.enable_anchoring:
            self.register_buffer("anchors", means.detach().clone())
            self.anchor_distance_scaled = self.config.anchor_distance * scene_scale
        else:
            self.anchors = None
            self.anchor_distance_scaled = None

        if self.config.enable_scale_clamp:
            self.register_buffer("anchor_scales", scales.detach().clone())
        else:
            self.anchor_scales = None

        # Strategy
        self.strategy = DMVStrategy(
            prune_alpha_thresh=self.config.prune_alpha_thresh,
            prune_every=self.config.prune_every,
        )

        # Metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # State
        self.step = 0
        if self.config.background_color == "random":
            self.background_color = torch.tensor([0.1490, 0.1647, 0.2157])
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def num_points(self):
        return self.gauss_params["means"].shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):
        self.step = 30000
        if "means" in dict:
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                return torch.rand(3, device=self.device)
            else:
                return self.background_color.to(self.device)
        elif self.config.background_color == "white":
            return torch.ones(3, device=self.device)
        else:
            return torch.zeros(3, device=self.device)

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        if not isinstance(camera, Cameras):
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"

        viewmat = get_viewmat(camera.camera_to_worlds)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())

        colors = torch.cat((self.features_dc[:, None, :], self.features_rest), dim=1)

        if self.config.sh_degree > 0:
            sh_degree_to_use = self.config.sh_degree
        else:
            colors = torch.sigmoid(colors).squeeze(1)
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=self.means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB+ED",
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
        )

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        depth_im = render[:, ..., 3:4]
        depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_im,
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

    def get_gt_img(self, image: torch.Tensor):
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return image.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        metrics_dict = {
            "psnr": self.psnr(predicted_rgb, gt_rgb),
            "gaussian_count": self.num_points,
        }
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        if "mask" in batch:
            mask = batch["mask"].to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
        }

        # Depth loss
        if self.config.depth_regularize and "depth_image" in batch and "depth" in outputs:
            gt_depth = batch["depth_image"].to(self.device)
            pred_depth = outputs["depth"]
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)
            depth_loss = compute_depth_loss(pred_depth, gt_depth, mask)
            loss_dict["depth_loss"] = self.config.depth_loss_weight * depth_loss

        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box=None) -> Dict[str, torch.Tensor]:
        """Get outputs for a camera (used during evaluation)."""
        assert camera is not None, "must provide camera to gaussian model"
        return self.get_outputs(camera.to(self.device))

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def step_cb(step):
            self.step = step

        return [
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                step_cb,
            ),
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            ),
        ]

    def step_post_backward(self, step: int):
        assert step == self.step
        self.optimizers = self.kwargs.get("optimizers", {})

        with torch.no_grad():
            # Enforce anchor constraint
            if self.config.enable_anchoring and self.anchors is not None:
                clamped_means = enforce_anchor_constraint(
                    self.gauss_params["means"].data,
                    self.anchors,
                    self.anchor_distance_scaled,
                )
                self.gauss_params["means"].data.copy_(clamped_means)

            # Enforce scale constraint
            if self.config.enable_scale_clamp and self.anchor_scales is not None:
                clamped_scales = enforce_scale_constraint(
                    self.gauss_params["scales"].data,
                    self.anchor_scales,
                    self.config.max_scale_factor,
                )
                self.gauss_params["scales"].data.copy_(clamped_scales)

        # Optional pruning
        prune_mask = self.strategy.step_post_backward(
            params=self.gauss_params,
            optimizers=self.optimizers,
            step=step,
            pruning_enable=self.config.pruning_enable,
        )

        # Update anchors if pruning occurred
        if prune_mask is not None:
            if self.anchors is not None:
                self.anchors = self.anchors[prune_mask]
            if self.anchor_scales is not None:
                self.anchor_scales = self.anchor_scales[prune_mask]

        # Log histograms
        if step > 0 and step % self.config.histogram_log_every == 0:
            self._log_histograms(step)

    def _log_histograms(self, step: int):
        try:
            import wandb

            if wandb.run is None:
                return

            with torch.no_grad():
                # Histogram of distances from anchors
                if self.anchors is not None:
                    distances = (self.gauss_params["means"] - self.anchors).norm(dim=-1)
                    wandb.log({"dmv/anchor_distances": wandb.Histogram(distances.cpu())}, step=step)

                # Histogram of scales (real space)
                scales_real = torch.exp(self.gauss_params["scales"])
                wandb.log({"dmv/scales": wandb.Histogram(scales_real.cpu())}, step=step)

                # Histogram of scale ratios vs initial
                if self.anchor_scales is not None:
                    scale_ratios = torch.exp(self.gauss_params["scales"] - self.anchor_scales)
                    wandb.log({"dmv/scale_ratios": wandb.Histogram(scale_ratios.cpu())}, step=step)

        except ImportError:
            pass

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),
            "lpips": float(lpips),
        }
        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
