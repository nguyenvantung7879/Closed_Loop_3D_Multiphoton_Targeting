from __future__ import annotations

import torch
from torch import nn


def make_h1_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 1e-4,
    betas: tuple[float, float] = (0.75, 0.95),
) -> torch.optim.Adam:
    """Create the Adam optimizer used by the H1 experiments."""
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The model has no trainable parameters")
    return torch.optim.Adam(parameters, lr=learning_rate, betas=betas)


def _require_connected_modules(model: nn.Module) -> None:
    for name in ("h1", "h2", "fusion"):
        if not hasattr(model, name):
            raise TypeError(f"Connected model is missing its {name!r} module")


def _split_h2_parameters(
    model: nn.Module,
    *,
    detail_learning_rate: float | None,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Separate the optional H2 detail head from the remaining H2 weights."""
    detail_module = getattr(model.h2, "additive_detail_head", None)
    if detail_learning_rate is not None and detail_module is None:
        raise ValueError("H2 has no enabled additive detail head")
    detail_parameters = (
        [
            parameter
            for parameter in detail_module.parameters()
            if parameter.requires_grad
        ]
        if detail_learning_rate is not None and detail_module is not None
        else []
    )
    detail_parameter_ids = {id(parameter) for parameter in detail_parameters}
    h2_parameters = [
        parameter
        for parameter in model.h2.parameters()
        if parameter.requires_grad and id(parameter) not in detail_parameter_ids
    ]
    return h2_parameters, detail_parameters


def make_h2_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 1e-4,
    detail_learning_rate: float | None = None,
    betas: tuple[float, float] = (0.75, 0.95),
) -> torch.optim.Adam:
    """Optimize H2 alone; H1 and fusion are required to be frozen."""
    if learning_rate <= 0:
        raise ValueError("H2 learning rate must be positive")
    if detail_learning_rate is not None and detail_learning_rate <= 0:
        raise ValueError("H2 detail learning rate must be positive")
    _require_connected_modules(model)
    if any(parameter.requires_grad for parameter in model.h1.parameters()):
        raise ValueError("Freeze H1 before constructing the H2 optimizer")
    if any(parameter.requires_grad for parameter in model.fusion.parameters()):
        raise ValueError("Freeze fusion before constructing the H2 optimizer")
    h2_parameters, detail_parameters = _split_h2_parameters(
        model, detail_learning_rate=detail_learning_rate
    )
    if not h2_parameters and not detail_parameters:
        raise ValueError("H2 must have trainable parameters")
    parameter_groups = [
        {"params": h2_parameters, "lr": learning_rate, "name": "h2"}
    ]
    if detail_parameters:
        parameter_groups.append(
            {
                "params": detail_parameters,
                "lr": detail_learning_rate,
                "name": "h2_detail",
            }
        )
    return torch.optim.Adam(
        parameter_groups,
        betas=betas,
    )


def make_fusion_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 2e-5,
    betas: tuple[float, float] = (0.75, 0.95),
) -> torch.optim.Adam:
    """Optimize fusion alone; both reconstruction stages must be frozen."""
    if learning_rate <= 0:
        raise ValueError("Fusion learning rate must be positive")
    _require_connected_modules(model)
    if any(parameter.requires_grad for parameter in model.h1.parameters()):
        raise ValueError("Freeze H1 before constructing the fusion optimizer")
    if any(parameter.requires_grad for parameter in model.h2.parameters()):
        raise ValueError("Freeze H2 before constructing the fusion optimizer")
    fusion_parameters = [
        parameter
        for parameter in model.fusion.parameters()
        if parameter.requires_grad
    ]
    if not fusion_parameters:
        raise ValueError("Fusion must have trainable parameters")
    return torch.optim.Adam(
        (
            {
                "params": fusion_parameters,
                "lr": learning_rate,
                "name": "fusion",
            },
        ),
        betas=betas,
    )


def make_h2_fusion_optimizer(
    model: nn.Module,
    *,
    h2_learning_rate: float = 1e-4,
    h2_detail_learning_rate: float | None = None,
    fusion_learning_rate: float = 1e-4,
    betas: tuple[float, float] = (0.75, 0.95),
) -> torch.optim.Adam:
    """Create separate H2/fusion parameter groups with H1 required frozen."""
    if h2_learning_rate <= 0 or fusion_learning_rate <= 0:
        raise ValueError("H2 and fusion learning rates must be positive")
    if h2_detail_learning_rate is not None and h2_detail_learning_rate <= 0:
        raise ValueError("H2 detail learning rate must be positive")
    _require_connected_modules(model)
    if any(parameter.requires_grad for parameter in model.h1.parameters()):
        raise ValueError("Freeze H1 before constructing the H2+fusion optimizer")
    h2_parameters, detail_parameters = _split_h2_parameters(
        model, detail_learning_rate=h2_detail_learning_rate
    )
    fusion_parameters = [
        parameter
        for parameter in model.fusion.parameters()
        if parameter.requires_grad
    ]
    if not h2_parameters or not fusion_parameters:
        raise ValueError("Both H2 and fusion must be trainable")
    parameter_groups = [
        {"params": h2_parameters, "lr": h2_learning_rate, "name": "h2"}
    ]
    if detail_parameters:
        parameter_groups.append(
            {
                "params": detail_parameters,
                "lr": h2_detail_learning_rate,
                "name": "h2_detail",
            }
        )
    parameter_groups.append(
        {
            "params": fusion_parameters,
            "lr": fusion_learning_rate,
            "name": "fusion",
        }
    )
    return torch.optim.Adam(parameter_groups, betas=betas)


def make_joint_optimizer(
    model: nn.Module,
    *,
    h1_learning_rate: float = 1e-5,
    h2_learning_rate: float = 2e-5,
    h2_detail_learning_rate: float | None = None,
    fusion_learning_rate: float = 2e-5,
    betas: tuple[float, float] = (0.75, 0.95),
) -> torch.optim.Adam:
    """Optimize the complete learned H1/H2/fusion path with separate rates.

    The connected model must first be configured for its ``joint`` stage.
    Parameters intentionally kept fixed by a module, such as H1's physical
    initial-estimate operator, remain outside every optimizer group.
    """
    learning_rates = {
        "H1": h1_learning_rate,
        "H2": h2_learning_rate,
        "fusion": fusion_learning_rate,
    }
    if any(value <= 0 for value in learning_rates.values()):
        raise ValueError("H1, H2, and fusion learning rates must be positive")
    if h2_detail_learning_rate is not None and h2_detail_learning_rate <= 0:
        raise ValueError("H2 detail learning rate must be positive")
    _require_connected_modules(model)

    h1_parameters = [
        parameter for parameter in model.h1.parameters() if parameter.requires_grad
    ]
    h2_parameters, detail_parameters = _split_h2_parameters(
        model, detail_learning_rate=h2_detail_learning_rate
    )
    fusion_parameters = [
        parameter
        for parameter in model.fusion.parameters()
        if parameter.requires_grad
    ]
    if not h1_parameters or not h2_parameters or not fusion_parameters:
        raise ValueError(
            "H1, H2, and fusion must all be trainable; configure the joint stage "
            "before constructing its optimizer"
        )

    parameter_groups = [
        {"params": h1_parameters, "lr": h1_learning_rate, "name": "h1"},
        {"params": h2_parameters, "lr": h2_learning_rate, "name": "h2"},
    ]
    if detail_parameters:
        parameter_groups.append(
            {
                "params": detail_parameters,
                "lr": h2_detail_learning_rate,
                "name": "h2_detail",
            }
        )
    parameter_groups.append(
        {
            "params": fusion_parameters,
            "lr": fusion_learning_rate,
            "name": "fusion",
        }
    )
    return torch.optim.Adam(parameter_groups, betas=betas)


def make_time_decay_scheduler(
    optimizer: torch.optim.Optimizer, *, decay: float = 0.01
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda completed_epochs: 1.0 / (1.0 + decay * completed_epochs),
    )
