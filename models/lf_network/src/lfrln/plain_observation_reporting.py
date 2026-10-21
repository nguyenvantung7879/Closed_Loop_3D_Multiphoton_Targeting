"""Display wording only; mathematical operators and metric keys are unchanged."""
from .evaluation_labels import label_methods
from . import context_reporting as original


def tf_comparison(*args,**kwargs):
    record,figures=original.tf_comparison(*args,**kwargs)
    return record,{key:label_methods(figure) for key,figure in figures.items()}


def plot_volume_columns(*args,**kwargs):
    return label_methods(original.plot_volume_columns(*args,**kwargs))


def stage_figure(*args,**kwargs):
    return label_methods(original.stage_figure(*args,**kwargs))


def tf_summary(label,record):
    names={'source21':'Source network (notebook 21)','interrupted22':'Earlier notebook 22 run',
           'current_branch':'Current network'}
    parts=[names.get(label,label)]
    for key,title in [('core_tf_shape_mse','TF agreement, reconstruction'),
                      ('context_tf_shape_mse','TF agreement, including surroundings')]:
        if key in record['metrics']:
            parts.append(f"{title}: {record['metrics'][key]['specimen_mean']:.4f}")
    return ' | '.join(parts)+' | lower is better; shape comparison only'
