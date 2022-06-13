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
import hacl.envs.gridworld.minigrid.minigrid_v20220407 as mg

io.set_fs_verbose()
pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

parser = jacinle.JacArgumentParser()
parser.add_argument('env', choices=['minigrid'])
parser.add_argument('task', choices=mg.SUPPORTED_TASKS)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--update-interval', type=int, default=50)
parser.add_argument('--print-interval', type=int, default=50)
parser.add_argument('--evaluate-interval', type=int, default=250)
parser.add_argument('--save-interval', type=int, default=250)
parser.add_argument('--iterations', type=int, default=10000)
parser.add_argument('--use-offline', type='bool', default=False)
parser.add_argument('--action-loss-weight', type=float, default=1.0)
parser.add_argument('--goal-loss-weight', type=float, default=1.0)

parser.add_argument('--action-mode', type=str, default='envaction', choices=mg.SUPPORTED_ACTION_MODES)
parser.add_argument('--structure-mode', type=str, default='full', choices=mg.SUPPORTED_STRUCTURE_MODES)
parser.add_argument('--train-heuristic', action='store_true')
parser.add_argument('--load', type='checked_file', default=None)
parser.add_argument('--load-heuristic', type='checked_file', default=None)
parser.add_argument('--discretize', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--visualize-failure', action='store_true', help='visualize the tasks that the planner failed to solve.')
parser.add_argument('--evaluate-objects', type=int, default=4)
parser.add_argument('--relevance-analysis', type='bool', default=True)
parser.add_argument('--heuristic', type=str, default='hff', choices=['hff', 'blind', 'external'])

# model-specific options.
parser.add_argument('--model-use-vq', type='bool', default=False, help='only for cw.ModelAbskin2')

# append the results
parser.add_argument('--append-expr', action='store_true')
parser.add_argument('--append-result', action='store_true')

# debug options for neural network training
parser.add_argument('--use-gt-classifier', type='bool', default=False)
parser.add_argument('--use-gt-facing', type='bool', default=False)

# debug options
parser.add_argument('--debug', action='store_true')
parser.add_argument('--debug-domain', action='store_true')
parser.add_argument('--debug-env', action='store_true')
parser.add_argument('--debug-encoder', action='store_true')
parser.add_argument('--debug-action', action='store_true', help='must be used with --debug-encoder')
parser.add_argument('--debug-strips', action='store_true')
parser.add_argument('--debug-ray', action='store_true')
parser.add_argument('--debug-discretize', action='store_true')
args = parser.parse_args()

if args.workers > 0:
    import ray
    ray.init()

if args.env == 'minigrid':
    lib = mg
    import hacl.p.kfac.minigrid.models as lib_models
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

    from hacl.p.kfac.minigrid.data_generator import worker_offline, worker_search
    if args.use_offline:
        worker_inner = worker_offline
    else:
        worker_inner = worker_search
else:
    raise ValueError(f'Unknown environment: {args.env}')

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

args.id_string = f'{args.task}-{args.action_mode}-{args.structure_mode}-lr={args.lr}-load={load_id}-{log_timestr}'
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


def build_heuristic_model(args, domain, **kwargs):
    HeuristicModel = lib_models.get_heuristic_model(args)
    return HeuristicModel(domain, **kwargs)


def worker(index, args, update_model_fn, push_data_fn):
    jacinle.reset_global_seed(index)

    logger.info(f'Worker {index} started. Arguments: {args}')
    logger.critical(f'Worker {index}: creating the domain: action mode = {args.action_mode} structure_mode = {args.structure_mode}')
    lib.set_domain_mode(args.action_mode, args.structure_mode)
    domain = lib.get_domain(force_reload=True)
    logger.critical(f'Worker {index}: creating model...')
    model = build_model(args, domain)
    if args.load is not None:
        load_state_dict(model, io.load(args.load))
    logger.critical(f'Worker {index}: creating environment (task={args.task})...')
    if args.env == 'minigrid':
        env = lib.make(args.task)
    else:
        raise ValueError(f'Unknown environment: {args.env}.')

    while True:
        data = worker_inner(args, domain, env, action_filter)
        push_data_fn(data)
        if not update_model_fn(model):
            break


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
    # for i in range(1024):
        states, actions, dones, goal, succ, extra_monitors = worker_inner(args, domain, env, action_filter)
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
    if args.env == 'minigrid' and args.structure_mode in ('abskin', 'full'):
        manual_feature_discretization = {
            'robot-direction': np.array([[0], [1], [2], [3]], dtype=np.int64),
            'robot-pose': np.stack(np.meshgrid(a, a), axis=-1).reshape(-1, 2),
            'item-pose': np.stack(np.meshgrid(a, a), axis=-1).reshape(-1, 2),
        }
    else:
        manual_feature_discretization = {}
    pds.ao_discretize(domain, dataset, {'robot-pose': 128, 'item-pose': 128, 'item-feature': 128, 'robot-direction': 16, 'location-pose': 128, 'location-feature': 128, 'robot-feature': 128}, cache_bool_features=True, manual_feature_discretizations=manual_feature_discretization)

    if args.debug_discretize:
        translator = pds.strips.GStripsTranslatorSAS(domain, cache_bool_features=True)
        for i in range(100):
            obs = env.reset()
            env.render()
            task = translator.compile_task(obs['state'], obs['mission'], forward_relevance_analysis=False, backward_relevance_analysis=True, verbose=True)
            relaxed = translator.recompile_relaxed_task(task, forward_relevance_analysis=False, backward_relevance_analysis=True)
            hff = pds.strips.StripsHFFHeuristic(relaxed, translator)
            print(hff.compute(task.state))
            QUIT = False
            from IPython import embed; embed()
            if QUIT:
                break


def evaluate(domain, env, heuristic_model=None, visualize: bool = False):
    def external_heuristic_function(state, goal):
        return heuristic_model.forward(state, goal)

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
            external_heuristic_function=external_heuristic_function,
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

        if args.load_heuristic is not None:
            discretize_str = 'learning'
        elif args.heuristic == 'hff':
            discretize_str = 'discretized' if args.discretize else 'no-discretized'
        else:
            assert args.heuristic == 'blind'
            discretize_str = 'blind'
        evaluate_key = f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects}-{discretize_str}'
        io.dump(f'./results/{evaluate_key}.json', nr_expansions_hist)
        return succ / total, nr_expansions / total


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

    heuristic_model = None
    if args.load_heuristic is not None:
        logger.critical('Creating the heuristic estimation model...')
        heuristic_model = build_heuristic_model(
            args,
            domain,
            predicates=['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey', 'is-key', 'is-ball', 'is-box', 'is-door', 'is-open'],
        )
        try:
            load_state_dict(heuristic_model, io.load(args.load_heuristic))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')

    # It will be used for evaluation.
    logger.critical(f'Creating environment (task={args.task})...')

    if args.env == 'minigrid':
        env = lib.make(args.task)
    else:
        raise ValueError(f'Unknown environment: {args.env}.')

    if args.debug_domain:
        domain.print_summary()
        return

    if args.debug_env:
        logger.critical('Visualizing...')
        env.reset()
        env.render()
        if hasattr(env, 'Actions'):
            for i in range(10):
                action = jacinle.random.choice_list(env.Actions)
                logger.info(f'  Taking action: {action}...')
                env.step_inner(action)
                env.render()
                time.sleep(0.5)
        return

    if args.debug_encoder:
        obs = env.reset()
        state = obs['state']
        domain.forward_augmented_features(state)

        if args.debug_action:
            for k, v in domain.operators.items():
                logger.critical('Executing {}...'.format(k))
                succ, state = v('r')(state)
                assert succ
                print(state)
                print('-' * 120)
        else:
            from IPython import embed; embed()
        return

    if args.debug_strips:
        logger.critical('Creating strips...')
        obs = env.reset()
        state = obs['state']
        goal_expr = obs['mission']
        actions = pds.generate_all_partially_grounded_actions(domain, state)

        from hacl.pdsketch.interface.v2.strips.grounding import GStripsTranslator
        strips_translator = GStripsTranslator(domain, use_string_name=True, prob_goal_threshold=0.99)
        strips_task = strips_translator.compile_task(state, goal_expr, actions, is_relaxed=False, relevance_analysis=args.relevance_analysis)
        print(strips_task)
        return

    if args.debug_ray:
        logger.critical('Debugging ray...')

        import hacl.p.kfac.mprl.workers as W

        nr_workers = 4
        data_store = W.DataStore.remote()
        data_workers = [W.DataWorker.remote(i, args, data_store, worker) for i in range(nr_workers)]

        print(data_workers)
        for i in range(100):
            while True:
                data = ray.get(data_store.pop.remote())
                if data is not None:
                    break
                time.sleep(0.1)
            states, actions, dones, goal, succ, extra_monitors = data
            print(i, succ, extra_monitors)
        return

    if args.evaluate:
        args.iterations = 0

    data_store = None
    data_workers = list()
    if args.workers > 0:
        logger.warning('Using {} workers.'.format(args.workers))
        import hacl.p.kfac.mprl.workers as W
        data_store = W.DataStore.remote()
        data_workers = [W.DataWorker.remote(i, args, data_store, worker) for i in range(args.workers)]
    else:
        logger.warning('Using single-threaded mode.')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    meters = jacinle.GroupMeters()
    for i in jacinle.tqdm(range(1, args.iterations + 1)):
        total_n, total_loss = 0, 0
        while total_n < 32:
            end = time.time()
            if args.workers > 0:
                while True:
                    data = ray.get(data_store.pop.remote())
                    if data is not None:
                        break
                    time.sleep(0.1)
                states, actions, dones, goal, succ, extra_monitors = data
            else:
                states, actions, dones, goal, succ, extra_monitors = worker_inner(args, domain, env, action_filter)

            extra_monitors['time/data'] = time.time() - end; end = time.time()
            loss, monitors, output_dict = model(feed_dict={'states': states, 'actions': actions, 'dones': dones, 'goal_expr': goal}, forward_augmented=True)
            extra_monitors['time/model'] = time.time() - end; end = time.time()
            extra_monitors['accuracy/succ'] = float(succ)
            monitors.update(extra_monitors)

            n = len(dones)
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

        if args.update_interval > 0 and i % args.update_interval == 0:
            if args.workers > 0:
                params = state_dict(model)
                for w in data_workers:
                    w.set_model_parameters.remote(params)

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
            ckpt_name = f'dumps/{args.structure_mode}-{args.task}-load={load_id}-epoch={i}.pth'
            logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
            io.dump(ckpt_name, state_dict(model))

    if args.discretize:
        logger.info('Discrete analysis...')
        discrete_analysis(domain, env)

    if args.evaluate:
        if args.env == 'minigrid':
            env.set_options(nr_objects=args.evaluate_objects)
        if not args.visualize:
            succ_rate, avg_expansions = evaluate(domain, env, heuristic_model=heuristic_model, visualize=False)
            if args.append_result:
                with open('./experiments-eval.txt', 'a') as f:
                    print(f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects},{succ_rate},{avg_expansions}', file=f)
            print(f'succ_rate = {succ_rate}')
            print(f'avg_expansions = {avg_expansions}')
        else:
            evaluate(domain, env, heuristic_model=heuristic_model, visualize=True)
    else:
        ckpt_name = f'dumps/{args.structure_mode}-{args.task}-load={load_id}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

    # from IPython import embed; embed()


def main_heuristics():
    jacinle.reset_global_seed(args.seed, True)
    lib.set_domain_mode(args.action_mode, args.structure_mode)
    domain = lib.get_domain()

    logger.critical('Creating model...')
    model = build_heuristic_model(
        args,
        domain,
        predicates=['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey', 'is-key', 'is-ball', 'is-box', 'is-door', 'is-open'],
    )
    if args.load is not None:
        try:
            load_state_dict(model, io.load(args.load))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')

    # It will be used for evaluation.
    logger.critical(f'Creating environment (task={args.task})...')

    if args.env == 'minigrid':
        env = lib.make(args.task)
    else:
        raise ValueError(f'Unknown environment: {args.env}.')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    meters = jacinle.GroupMeters()
    for i in jacinle.tqdm(range(1, args.iterations + 1)):
        total_n, total_loss = 0, 0
        while total_n < 32:
            end = time.time()
            states, actions, dones, goal, succ, extra_monitors = worker_inner(args, domain, env, action_filter)

            if not succ:
                continue

            extra_monitors['time/data'] = time.time() - end; end = time.time()
            loss, monitors, output_dict = model.bc(states, goal)
            extra_monitors['time/model'] = time.time() - end; end = time.time()
            extra_monitors['accuracy/succ'] = float(succ)
            monitors.update(extra_monitors)

            n = len(dones)
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

        if args.save_interval > 0 and i % args.save_interval == 0:
            ckpt_name = f'dumps/heuristic-{args.structure_mode}-{args.task}-load={load_id}-epoch={i}.pth'
            logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
            io.dump(ckpt_name, state_dict(model))

    ckpt_name = f'dumps/heuristic-{args.structure_mode}-{args.task}-load={load_id}.pth'
    logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
    io.dump(ckpt_name, state_dict(model))

    # from IPython import embed; embed()


if __name__ == '__main__':
    if args.train_heuristic:
        main_heuristics()
    else:
        main()
