from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses_h2 import H2FusionLossConfig, H2LossConfig, h2_fusion_loss, h2_loss


def _run_h2_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_function: Callable[..., dict[str, torch.Tensor]],
    loss_config: H2LossConfig | H2FusionLossConfig,
    required_training_stages: tuple[str, ...],
    prediction_function: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    gradient_clip_norm: float | None,
    max_batches: int | None,
    progress: bool,
    batch_history: list[dict[str, float | int]] | None,
    target_key: str,
) -> dict[str, float]:
    training = optimizer is not None
    if training and getattr(model, "active_stage", None) not in required_training_stages:
        expected = " or ".join(repr(stage) for stage in required_training_stages)
        raise ValueError(f"Configure the model for {expected} before training")
    model.train(training)
    totals: dict[str, float] = {}
    observations = 0

    batches: Any = loader
    if progress:
        from tqdm.auto import tqdm

        batches = tqdm(loader, desc="train" if training else "validation")

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            if target_key not in batch:
                raise ValueError(f"H2 batch is missing raw target key {target_key!r}")
            lf = batch["lf"].to(device, non_blocking=True)
            raw_target = batch[target_key].to(device, non_blocking=True)
            batch_size = lf.shape[0]

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                predictions = prediction_function(lf)
                losses = loss_function(*predictions, raw_target, config=loss_config)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses["total"]).backward()
                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip_norm
                        )
                    optimizer.step()

            for name, value in losses.items():
                totals.setdefault(name, 0.0)
                totals[name] += float(value.detach()) * batch_size
            if batch_history is not None:
                batch_history.append(
                    {
                        "batch": batch_index + 1,
                        "batch_size": batch_size,
                        **{
                            name: float(value.detach())
                            for name, value in losses.items()
                        },
                    }
                )
            observations += batch_size
            if progress:
                batches.set_postfix(total=f"{float(losses['total'].detach()):.5g}")

    if observations == 0:
        raise ValueError("The DataLoader produced no batches")
    return {name: value / observations for name, value in totals.items()}


def run_h2_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: H2LossConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
    progress: bool = False,
    batch_history: list[dict[str, float | int]] | None = None,
    target_key: str = "gt",
) -> dict[str, float]:
    """Run one H2-only epoch while H1 supplies a frozen initial volume."""
    if not callable(getattr(model, "forward_h2", None)):
        raise TypeError("H2-only training requires forward_h2()")

    def predictions(
        lf: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor | None, torch.Tensor | None,
    ]:
        result = model.forward_h2(lf, return_intermediates=True)
        if not isinstance(result, dict):  # pragma: no cover - type guard
            raise RuntimeError("H2 did not return its intermediate tensors")
        return (
            result["reconstruction"],
            result["attention_weights"],
            result.get("predicted_views"),
            result.get("observed_views"),
        )

    def loss_with_attention(
        h2_prediction: torch.Tensor,
        attention_weights: torch.Tensor,
        predicted_views: torch.Tensor | None,
        observed_views: torch.Tensor | None,
        raw_target: torch.Tensor,
        *,
        config: H2LossConfig,
    ) -> dict[str, torch.Tensor]:
        return h2_loss(
            h2_prediction,
            raw_target,
            config=config,
            attention_weights=attention_weights,
            predicted_views=predicted_views,
            observed_views=observed_views,
        )

    return _run_h2_epoch(
        model,
        loader,
        device=device,
        loss_function=loss_with_attention,
        loss_config=loss_config,
        required_training_stages=("h2",),
        prediction_function=predictions,
        optimizer=optimizer,
        scaler=scaler,
        use_amp=use_amp,
        gradient_clip_norm=gradient_clip_norm,
        max_batches=max_batches,
        progress=progress,
        batch_history=batch_history,
        target_key=target_key,
    )


def run_h2_fusion_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: H2FusionLossConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
    progress: bool = False,
    batch_history: list[dict[str, float | int]] | None = None,
    target_key: str = "gt",
) -> dict[str, float]:
    """Run one connected H2+fusion epoch against a raw-emitter target."""
    return _run_connected_fusion_epoch(
        model,
        loader,
        device=device,
        loss_config=loss_config,
        required_training_stages=("h2_fusion", "joint"),
        optimizer=optimizer,
        scaler=scaler,
        use_amp=use_amp,
        gradient_clip_norm=gradient_clip_norm,
        max_batches=max_batches,
        progress=progress,
        batch_history=batch_history,
        target_key=target_key,
    )


def run_fusion_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: H2FusionLossConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    gradient_clip_norm: float | None = None,
    max_batches: int | None = None,
    progress: bool = False,
    batch_history: list[dict[str, float | int]] | None = None,
    target_key: str = "gt",
) -> dict[str, float]:
    """Run one fusion-only epoch while H1 and H2 remain frozen."""
    return _run_connected_fusion_epoch(
        model,
        loader,
        device=device,
        loss_config=loss_config,
        required_training_stages=("fusion",),
        optimizer=optimizer,
        scaler=scaler,
        use_amp=use_amp,
        gradient_clip_norm=gradient_clip_norm,
        max_batches=max_batches,
        progress=progress,
        batch_history=batch_history,
        target_key=target_key,
    )


def _run_connected_fusion_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: H2FusionLossConfig,
    required_training_stages: tuple[str, ...],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    gradient_clip_norm: float | None,
    max_batches: int | None,
    progress: bool,
    batch_history: list[dict[str, float | int]] | None,
    target_key: str,
) -> dict[str, float]:
    if not callable(getattr(model, "predict_stages", None)):
        raise TypeError("H2+fusion training requires predict_stages()")

    def predictions(
        lf: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor | None, torch.Tensor | None,
    ]:
        stages = model.predict_stages(lf, return_loss_terms=True)
        return (
            stages["merged"],
            stages["h2"],
            stages["fusion_residual"],
            stages["h2_attention_weights"],
            stages.get("h2_predicted_views"),
            stages.get("h2_observed_views"),
        )

    def loss_with_residual(
        final_prediction: torch.Tensor,
        h2_prediction: torch.Tensor,
        fusion_residual: torch.Tensor,
        attention_weights: torch.Tensor,
        predicted_views: torch.Tensor | None,
        observed_views: torch.Tensor | None,
        raw_target: torch.Tensor,
        *,
        config: H2FusionLossConfig,
    ) -> dict[str, torch.Tensor]:
        return h2_fusion_loss(
            final_prediction,
            h2_prediction,
            raw_target,
            config=config,
            fusion_residual=fusion_residual,
            attention_weights=attention_weights,
            predicted_views=predicted_views,
            observed_views=observed_views,
        )

    return _run_h2_epoch(
        model,
        loader,
        device=device,
        loss_function=loss_with_residual,
        loss_config=loss_config,
        required_training_stages=required_training_stages,
        prediction_function=predictions,
        optimizer=optimizer,
        scaler=scaler,
        use_amp=use_amp,
        gradient_clip_norm=gradient_clip_norm,
        max_batches=max_batches,
        progress=progress,
        batch_history=batch_history,
        target_key=target_key,
    )
