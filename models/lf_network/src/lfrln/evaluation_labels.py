"""Unambiguous display names without changing historical method/cache IDs."""
from pathlib import Path

METHOD_LABELS = {
    '3d rl volume': '3D ISRA (least squares)',
    '3d rl measurement': '3D Poisson RL',
    'phase space rl': 'Phase-space RL',
    'network source': 'Source network',
    'network joint': 'Current network',
    'PS reference (filtered)': 'Measured PS stack',
    'Artificial emitter g': 'Known simulated object',
    'Artificial emitter': 'Known simulated object',
    'Final raw volume': 'Reconstruction',
    'Final raw prediction': 'Reconstruction',
    'T(final core), zero exterior': 'Predicted TF\nOutside volume assumed empty',
    'T(core + estimated halo)': 'Predicted TF\nWith surrounding fluorescence',
    'T(final prediction)': 'Predicted TF from reconstruction',
    'Corrected TF target T(GT)': 'TF simulated from known object',
    'Raw synthetic GT': 'Known simulated object',
}


def label_methods(figure):
    for axis in figure.axes:
        title = axis.get_title()
        if title in METHOD_LABELS:
            axis.set_title(METHOD_LABELS[title])
    if figure._suptitle is not None:
        title=figure._suptitle.get_text()
        title=title.replace('ps_shaped_artificial','Simulated LF from PS stack (artificial test object)')
        title=title.replace('measured_biology','Measured biological LF')
        figure._suptitle.set_text(title)
    return figure


def plot_case(path):
    from .reconstruction_benchmark import plot_case as original
    path = Path(path)
    figures = original(path)
    for figure, destination in zip(figures, (path.with_suffix('.png'), path.with_name(path.stem+'_lf.png'))):
        label_methods(figure).savefig(destination, dpi=110)
    return figures


def plot_fixed_slices(path):
    from .joint_reconstruction_evaluation import plot_fixed_slices as original
    path = Path(path)
    figure = label_methods(original(path))
    figure.savefig(path.with_name(path.stem+'_slices.png'), dpi=110)
    return figure
