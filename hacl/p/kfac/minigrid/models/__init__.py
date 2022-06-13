from .model_full import ModelFull
from .model_basic import ModelBasic
from .model_robokin import ModelRobokin
from .model_abskin import ModelAbskin, HeuristicModelAbskin


def get_model(args):
    if args.structure_mode == 'full':
        Model = ModelFull
    elif args.structure_mode == 'basic':
        Model = ModelBasic
        args.relevance_analysis = False
        args.heuristic = 'blind'
    elif args.structure_mode == 'robokin':
        Model = ModelRobokin
        args.relevance_analysis = False
        args.heuristic = 'blind'
    elif args.structure_mode == 'abskin':
        Model = ModelAbskin
    else:
        raise ValueError('Unknown structure mode: {}.'.format(args.structure_mode))

    return Model


def get_heuristic_model(args):
    if args.structure_mode == 'abskin':
        Model = HeuristicModelAbskin
    else:
        raise ValueError('Unknown structure mode: {}.'.format(args.structure_mode))
    return Model