import os.path as osp
import time
from copy import deepcopy

import numpy as np
import torch
import jacinle
import jacinle.io as io
import jactorch
from jactorch.io import load_state_dict, state_dict

import hacl.pdsketch as pds
import hacl.envs.gridworld.minigrid.minigrid_v20220407 as lib
import hacl.p.kfac.minigrid.models as lib_models
from hacl.p.kfac.minigrid.data_generator import worker_offline

io.set_fs_verbose()
pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

parser = jacinle.JacArgumentParser()
parser.add_argument('--task', default='goto2', choices=lib.SUPPORTED_TASKS)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--iterations', type=int, default=10000, help='number of gradient descent iterations')
parser.add_argument('--episodes', type=int, default=10000, choices=[100, 1000, 10000, 100000], help='the number of episodes in the training dataset')

parser.add_argument('--print-interval', type=int, default=50)
parser.add_argument('--evaluate-interval', type=int, default=250)
parser.add_argument('--save-interval', type=int, default=250)

parser.add_argument('--action-loss-weight', type=float, default=1.0)
parser.add_argument('--goal-loss-weight', type=float, default=1.0)

parser.add_argument('--action-mode', type=str, default='envaction', choices=lib.SUPPORTED_ACTION_MODES)
parser.add_argument('--structure-mode', type=str, default='full', choices=lib.SUPPORTED_STRUCTURE_MODES)
parser.add_argument('--load', type='checked_file', default=None)

parser.add_argument('--debug', action='store_true', help='run the script in the debug mode')

# evaluation-time related.
parser.add_argument('--generate-dataset', action='store_true')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
parser.add_argument('--discretize', action='store_true', help='discretize the latent representations')
parser.add_argument('--heuristic', type=str, default='hff', choices=['hff', 'blind', 'external'], help='heuristic for the evaluation')
parser.add_argument('--relevance-analysis', type='bool', default=True, help='whether to perform relevance analysis')
parser.add_argument('--visualize', action='store_true', help='visualize the evaluation results')
parser.add_argument('--visualize-failure', action='store_true', help='visualize the tasks that the planner failed to solve.')
parser.add_argument('--evaluate-objects', type=int, default=4, help='the number of objects to in the evaluation environment')

# append the results
parser.add_argument('--append-expr', action='store_true')
parser.add_argument('--append-result', action='store_true')

# debug options for neural network training
parser.add_argument('--use-gt-classifier', type='bool', default=False)
parser.add_argument('--use-gt-facing', type='bool', default=False)

args = parser.parse_args()
args.env = 'minigrid'

if args.task in ('goto', 'goto2', 'gotosingle'):
    def action_filter(a):
        return a.name in ['move', 'forward', 'lturn', 'rturn']
elif args.task == 'pickup' or args.task == 'generalization':
    def action_filter(a):
        return a.name in ['move', 'pickup', 'forward', 'lturn', 'rturn']
elif args.task == 'open':
    def action_filter(a):
        return a.name in ['move', 'pickup', 'forward', 'lturn', 'rturn', 'toggle']
else:
    raise ValueError(f'Unknown task: {args.task}.')

Model = lib_models.get_model(args)

assert args.task in lib.SUPPORTED_TASKS
assert args.action_mode in lib.SUPPORTED_ACTION_MODES
assert args.structure_mode in lib.SUPPORTED_STRUCTURE_MODES

logger = jacinle.get_logger(__file__)
log_timestr = time.strftime(time.strftime('%Y-%m-%d-%H-%M-%S'))
if args.load is not None:
    load_id = osp.basename(args.load).split('-')
    if load_id[1] == 'goto' and load_id[2] == 'single':
        load_id = 'gotosingle'
    else:
        load_id = load_id[1]
else:
    load_id = 'scratch'

args.id_string = f'{args.task}-{args.action_mode}-{args.structure_mode}-lr={args.lr}-load={load_id}-episodes={args.episodes}-{log_timestr}'
args.log_filename = f'dumps/{args.id_string}.log'
args.json_filename = f'dumps/{args.id_string}.jsonl'

if not args.debug:
    jacinle.set_logger_output_file(args.log_filename)

logger.critical('Arguments: {}'.format(args))

if not args.debug and args.append_expr:
    with open('./experiments.txt', 'a') as f:
        f.write(args.id_string + '\n')


def build_model(args, domain, **kwargs):
    return Model(domain, **kwargs)


def discretize_feature(domain: pds.Domain, dataset: pds.AODiscretizationDataset, feature_name, input_name=None):
    if input_name is None:
        input_name = feature_name
    mappings = dict()
    for state in dataset.iter_all_states():
        s = state.clone()
        domain.forward_features_and_axioms(s, forward_augmented=True, forward_derived=False, forward_axioms=False)
        input_feature = s[input_name].tensor.reshape(-1, s[input_name].tensor.shape[-1])
        output_feature = s[feature_name].tensor.reshape(-1, s[feature_name].tensor.shape[-1])

        assert input_feature.shape[0] == output_feature.shape[0]
        for i in range(input_feature.shape[0]):
            key = tuple(map(int, input_feature[i]))
            if key not in mappings:
                mappings[key] = output_feature[i]

    return jactorch.as_numpy(torch.stack([mappings[k] for k in sorted(mappings)], dim=0))


def discrete_analysis(domain, env):
    dataset = pds.AODiscretizationDataset()
    for i in range(1024):
        states, actions, dones, goal, succ, extra_monitors = worker_offline(args, domain, env, action_filter)
        dataset.feed(states, actions, dones, goal)

        # if succ:
        #     state = states[-2]
        #     state = state.clone()
        #     domain.forward_features_and_axioms(state)
        #     action: pds.OperatorApplier = actions[-1]
        #     next_state = action.apply_effect(state)
        #     print(action)
        #     print(next_state['item-feature'].tensor[-3])
        #     import ipdb; ipdb.set_trace()

    a = np.arange(8)
    manual_feature_discretization = {
        'robot-direction': np.array([[0], [1], [2], [3]], dtype=np.int64),
        'robot-pose': np.stack(np.meshgrid(a, a), axis=-1).reshape(-1, 2),
        'item-pose': np.stack(np.meshgrid(a, a), axis=-1).reshape(-1, 2),
    }
    pds.ao_discretize(domain, dataset, {'robot-pose': 128, 'item-pose': 128, 'item-feature': 128, 'robot-direction': 16, 'location-pose': 128, 'location-feature': 128}, cache_bool_features=True, manual_feature_discretizations=manual_feature_discretization)


def evaluate(domain, env, visualize: bool = False):
    def planner(state, mission):
        # pds.heuristic_search_strips.DEBUG = True
        args.relevance_analysis = True
        return pds.heuristic_search_strips(
            domain,
            state,
            mission,
            max_depth=10, max_expansions=500,
            action_filter=action_filter,
            use_tuple_desc=False,
            use_quantized_state=False,
            forward_augmented=True,
            track_most_promising_trajectory=True,
            strips_heuristic=args.heuristic,
            strips_forward_relevance_analysis=False,
            strips_backward_relevance_analysis=args.relevance_analysis,
            strips_use_sas=args.discretize,
            use_strips_op=False,
            verbose=True,
            return_extra_info=not visualize,
        )
    if visualize:
        lib.visualize_planner(env, planner)
    else:
        jacinle.reset_global_seed(args.seed, True)
        succ, nr_expansions, total = 0, 0, 0
        nr_expansions_hist = list()
        for i in jacinle.tqdm(100, desc='Evaluation'):
            obs = env.reset()
            env_copy = deepcopy(env)
            state, goal = obs['state'], obs['mission']
            plan, extra_info = planner(state, goal)
            this_succ = False
            if plan is None:
                pass
            else:
                for action in plan:
                    _, _, done, _ = env.step(action)
                    if done:
                        this_succ = True
                        break

            nr_expansions += extra_info['nr_expansions']
            succ += int(this_succ)
            total += 1

            print(this_succ, extra_info['nr_expansions'])
            nr_expansions_hist.append({'succ': this_succ, 'expansions': extra_info['nr_expansions']})

            if not this_succ and args.visualize_failure:
                print('Failed to solve the problem.')
                lib.visualize_plan(env_copy, plan=plan)

        # import seaborn as sns
        # sns.histplot(nr_expansions_hist)
        # import matplotlib.pyplot as plt
        # plt.savefig('./visualizations/nr_expansions_hist.png')

        if args.heuristic == 'hff':
            discretize_str = 'discretized' if args.discretize else 'no-discretized'
        else:
            assert args.heuristic == 'blind'
            discretize_str = 'blind'
        evaluate_key = f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects}-{discretize_str}'
        io.dump(f'./results/{evaluate_key}.json', nr_expansions_hist)
        return succ / total, nr_expansions / total


def create_dataset(args, domain):
    dataset = list()

    env = lib.make('gotosingle')
    nr_single_examples = int(0.25 * args.episodes)
    for _ in jacinle.tqdm(nr_single_examples, desc='Creating dataset for gotosingle'):
        state, actions, dones, goal, succ, extra_monitors = worker_offline(args, domain, env, action_filter)
        dataset.append({'states': state, 'actions': actions, 'dones': dones, 'goal_expr': goal, 'succ': succ})
    env = lib.make(args.task)
    for _ in jacinle.tqdm(args.episodes - nr_single_examples, desc=f'Creating dataset for {args.task}'):
        state, actions, dones, goal, succ, extra_monitors = worker_offline(args, domain, env, action_filter)
        dataset.append({'states': state, 'actions': actions, 'dones': dones, 'goal_expr': goal, 'succ': succ})
    return dataset


def main():
    jacinle.reset_global_seed(args.seed, True)
    lib.set_domain_mode(args.action_mode, args.structure_mode)
    domain = lib.get_domain()

    logger.critical('Creating model...')
    model = build_model(
        args,
        domain,
        goal_loss_weight=args.goal_loss_weight,
        action_loss_weight=args.action_loss_weight
    )
    if args.load is not None:
        try:
            load_state_dict(model, io.load(args.load))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')

    # It will be used for evaluation.
    logger.critical(f'Creating environment (task={args.task})...')
    env = lib.make(args.task)

    args.dataset_file = osp.join('data', args.env, f'{args.structure_mode}-{args.task}-episodes={args.episodes}-seed={args.seed}.pkl')
    if osp.isfile(args.dataset_file):
        logger.critical(f'Loading dataset from {args.dataset_file}...')
        dataset = io.load(args.dataset_file)
    else:
        logger.critical(f'Creating dataset for {args.dataset_file}...')
        jacinle.reset_global_seed(args.seed, True)
        io.mkdir(osp.dirname(args.dataset_file))
        dataset = create_dataset(args, domain)
        io.dump(args.dataset_file, dataset)

    if args.generate_dataset:
        logger.critical('Dataset generation finished. So quit the program.')
        return

    if args.evaluate:
        args.iterations = 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    meters = jacinle.GroupMeters()
    for i in jacinle.tqdm(range(1, args.iterations + 1)):
        total_n, total_loss = 0, 0
        while total_n < 32:
            end = time.time()
            extra_monitors = dict()
            data_index = np.random.randint(len(dataset))
            feed_dict = dataset[data_index]
            extra_monitors['time/data'] = time.time() - end; end = time.time()
            loss, monitors, output_dict = model(feed_dict=feed_dict, forward_augmented=True)
            extra_monitors['time/model'] = time.time() - end; end = time.time()
            monitors.update(extra_monitors)

            n = len(feed_dict['states'])
            total_n += n
            total_loss += loss * n

            meters.update(monitors, n=n)
            jacinle.get_current_tqdm().set_description(
                meters.format_simple(values={k: v for k, v in meters.val.items() if k.count('/') <= 1})
            )

        end = time.time()
        if total_loss.requires_grad:
            opt.zero_grad()
            total_loss /= total_n
            total_loss.backward()
            opt.step()
        meters.update({'time/gd': time.time() - end}, n=1)

        if args.print_interval > 0 and i % args.print_interval == 0:
            logger.info(meters.format_simple(f'Iteration {i}/{args.iterations}', values='avg', compressed=False))

            if not args.debug:
                with open(args.json_filename, 'a') as f:
                    f.write(jacinle.io.dumps_json({k: float(v) for k, v in meters.avg.items()}) + '\n')

            meters.reset()

        if args.evaluate_interval > 0 and i % args.evaluate_interval == 0:
            succ_rate = evaluate(domain, env, visualize=False)
            logger.critical(f'Iteration {i}/{args.iterations}: succ rate: {succ_rate}')

        if args.save_interval > 0 and i % args.save_interval == 0:
            ckpt_name = f'dumps/{args.structure_mode}-{args.task}-load={load_id}-episodes={args.episodes}-epoch={i}.pth'
            logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
            io.dump(ckpt_name, state_dict(model))

    if args.discretize:
        logger.info('Discrete analysis...')
        discrete_analysis(domain, env)

    if args.evaluate:
        pass
    else:
        ckpt_name = f'dumps/{args.structure_mode}-{args.task}-load={load_id}-episodes={args.episodes}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

    if True:  # always do the evaluation.
        if args.env == 'minigrid':
            env.set_options(nr_objects=args.evaluate_objects)
        if not args.visualize:
            succ_rate, avg_expansions = evaluate(domain, env, visualize=False)
            if args.append_result:
                with open('./experiments-eval.txt', 'a') as f:
                    print(f'{args.structure_mode}-{args.task}-load={load_id}-episodes={args.episodes}-objs={args.evaluate_objects},{succ_rate},{avg_expansions}', file=f)
            print(f'succ_rate = {succ_rate}')
            print(f'avg_expansions = {avg_expansions}')
        else:
            evaluate(domain, env, visualize=True)
    # from IPython import embed; embed()


if __name__ == '__main__':
    main()

