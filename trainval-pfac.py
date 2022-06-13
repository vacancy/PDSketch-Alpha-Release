import os.path as osp
import time
from copy import deepcopy

import numpy as np
import torchshow as ts
import torch
import jacinle
import jacinle.io as io
import jactorch
from jactorch.io import load_state_dict, state_dict

import hacl.pdsketch as pds
import hacl.p.kfac.pfac.model as lib

pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

parser = jacinle.JacArgumentParser()
parser.add_argument('--data-dir', type='checked_dir', required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--update-interval', type=int, default=50)
parser.add_argument('--print-interval', type=int, default=50)
parser.add_argument('--evaluate-interval', type=int, default=250)
parser.add_argument('--save-interval', type=int, default=250)
parser.add_argument('--iterations', type=int, default=10000)
parser.add_argument('--use-dense-supervision', type='bool', default='yes')
parser.add_argument('--use-goal-only', type='bool', default='no')

parser.add_argument('--load', type='checked_file', default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--visualize-failure', action='store_true', help='visualize the tasks that the planner failed to solve.')
parser.add_argument('--evaluate-objects', type=int, default=4)

# append the results
parser.add_argument('--append-expr', action='store_true')
parser.add_argument('--append-result', action='store_true')

parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

logger = jacinle.get_logger(__file__)
log_timestr = time.strftime(time.strftime('%Y-%m-%d-%H-%M-%S'))
if args.load is not None:
    load_id = osp.basename(args.load).split('-')
    if load_id[1] == 'goto' and load_id[2] == 'single':
        load_id = 'goto-single'
    else:
        load_id = load_id[1]
else:
    load_id = 'scratch'

args.id_string = f'pfac-lr={args.lr}-load={load_id}-goal_only={args.use_goal_only}-{log_timestr}'
args.log_filename = f'dumps/{args.id_string}.log'
args.json_filename = f'dumps/{args.id_string}.jsonl'

if not args.debug:
    jacinle.set_logger_output_file(args.log_filename)

logger.critical('Arguments: {}'.format(args))


def get_data():
    files = io.lsdir(args.data_dir, '*.pkl')
    files = [io.load(f) for f in files]
    return files


def process_image(img):
    return torch.tensor(img, dtype=torch.float32).permute((2, 0, 1)).flatten() / 255


def process_data(domain, data):
    succ = data['episode_info']['succ']
    colors = list(data['episode_info']['goal_color'])
    relation = data['episode_info']['goal_relation']
    object_ids = data['episode_info']['object_ids']

    states = list()
    actions = list()

    if args.use_dense_supervision:
        dense_goals = list()
        dense_dones = list()

        for color in colors:
            dense_goals.append(domain.parse('(exists (?o - item) (is-{} ?o))'.format(color)))
            dense_dones.append(list())
        subj = 0
        dense_goals.append(domain.parse(f'(exists (?o1 - item) (exists (?o2 - container) (and (is-{colors[subj]} ?o1) (is-target ?o2) (is-in ?o1 ?o2) ) ))'))
        dense_dones.append(list())
        for subj, obj, rel in relation:
            rel = rel.split()[0]
            dense_goals.append(domain.parse(
                f'(exists (?o1 - item) (exists(?o2 - item) (and (is-{colors[subj]} ?o1)(is-{colors[obj]} ?o2) (is-{rel} ?o1 ?o2) ) ))'
            ))
            dense_dones.append(list())
    else:
        dense_goals = list()
        dense_dones = list()

    data['episode'] = data['episode'][:3]
    data['episode'] = data['episode']
    for i, (rgb, eobs, action, reward, info, summary) in enumerate(data['episode']):
        object_names = list()
        object_types = list()
        item_images = list()
        item_poses = list()
        container_images = list()
        container_poses = list()
        for name in sorted(list(eobs.keys()), key=lambda name: int(name.split('#')[-1])):
            obj_id = int(name.split('#')[1])
            if obj_id == object_ids['clean_box'] or obj_id == object_ids['dry_box']:
                continue

            obj = eobs[name]
            object_names.append(name)
            object_types.append('item' if obj_id in object_ids['blocks'] else 'container')

            if object_types[-1] == 'item':
                item_images.append(process_image(obj['crop']))
                item_poses.append(torch.tensor(obj['pose'][0]))
            else:
                container_images.append(process_image(obj['crop']))
                container_poses.append(torch.tensor(obj['pose'][0]))
        state = pds.State([domain.types[t] for t in object_types], pds.ValueDict(), object_names)
        ctx = state.define_context(domain)
        ctx.define_feature('item-pose', torch.stack(item_poses).float())
        ctx.define_feature('container-pose', torch.stack(container_poses).float())
        ctx.define_feature('item-image', torch.stack(item_images))
        ctx.define_feature('container-image', torch.stack(container_images))
        states.append(state)

        if i != len(data['episode']) - 1:
            object_id = action['obj_id']
            container_id = action['target_container']
            if container_id == object_ids['target_box']:
                action = domain.operators['move-to'](f'item#{object_id}', pds.Value(domain.types['pose'], [], torch.tensor(
                    action['pose1'][0], dtype=torch.float32
                )))
            else:
                action = domain.operators['move-into'](f'item#{object_id}', f'item#{container_id}')
            actions.append(action)

        if args.use_dense_supervision:
            index = 0
            for color in colors:
                if color in summary['colors']:
                    dense_dones[index].append(1)
                else:
                    dense_dones[index].append(0)
                index += 1
            subj = 0
            found = False
            for k in range(3):
                if summary['colors'][k] == colors[subj] and summary['in_zone'][k]:
                    found = True
                    break
            dense_dones[index].append(int(found))
            index += 1
            for subj, obj, rel in relation:
                found = False
                for k1 in range(3):
                    for k2 in range(3):
                        if k1 == k2:
                            continue
                        if summary['colors'][k1] == colors[subj] and summary['colors'][k2] == colors[obj] and summary['in_zone'][k1] and summary['in_zone'][k2] and summary['relations'][k1][k2][rel]:
                            found = True
                            break
                    if found:
                        break
                dense_dones[index].append(int(found))
                index += 1

    dense_dones = [torch.tensor(dones_, dtype=torch.int64) for dones_ in dense_dones]
    # dones = torch.stack(dense_dones[-3:], dim=-1).min(dim=-1)[0]
    # goal = domain.parse(f"""(and
    #     (exists (?o1 - item) (exists (?o2 - container) (and (is-{colors[0]} ?o1) (is-target ?o2) (is-in ?o1 ?o2) ) ))
    #     (exists (?o1 - item) (exists (?o2 - item) (and (is-{colors[1]} ?o1) (is-{colors[relation[0][1]]} ?o2) (is-{relation[0][2].split()[0]} ?o1 ?o2) ) ))
    #     (exists (?o1 - item) (exists (?o2 - item) (and (is-{colors[2]} ?o1) (is-{colors[relation[1][1]]} ?o2) (is-{relation[1][2].split()[0]} ?o1 ?o2) ) ))
    # )""")
    dones = None
    goal = None

    extra_monitors = dict()
    # import ipdb; ipdb.set_trace()

    return states, actions, dones, goal, succ, extra_monitors, dense_goals, dense_dones, data


def main():
    jacinle.reset_global_seed(args.seed, True)
    domain = lib.load_domain()

    dataset = get_data()
    dataset = [process_data(domain, data) for data in dataset]

    # It will be used for evaluation.
    logger.critical('Creating model...')
    model = lib.ModelPFac(domain, action_loss_weight=0 if args.use_goal_only else 0.1)

    if args.load is not None:
        try:
            load_state_dict(model, io.load(args.load))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')

    if args.evaluate:
        args.iterations = 0

    logger.warning('Using single-threaded mode.')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    meters = jacinle.GroupMeters()
    for i in jacinle.tqdm(range(1, args.iterations + 1)):
        total_n, total_loss = 0, 0
        while total_n < 8:
            end = time.time()

            index = np.random.randint(0, len(dataset))
            states, actions, dones, goal, succ, extra_monitors, dense_goals, dense_dones, _ = dataset[index]

            if not succ:
                continue

            extra_monitors['time/data'] = time.time() - end; end = time.time()
            loss, monitors, output_dict = model(feed_dict={'states': states, 'actions': actions, 'dones': dones, 'goal_expr': goal, 'subgoals': dense_goals, 'subgoals_done': dense_dones}, forward_augmented=True)

            extra_monitors['time/model'] = time.time() - end; end = time.time()
            extra_monitors['accuracy/succ'] = float(succ)
            monitors.update(extra_monitors)

            n = len(states)
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

        if args.save_interval > 0 and i % args.save_interval == 0:
            ckpt_name = f'dumps/pfac-load={load_id}-epoch={i}.pth'
            logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
            io.dump(ckpt_name, state_dict(model))

    if args.evaluate:
        with torch.no_grad():
            for i in jacinle.tqdm(range(1, 100 + 1)):
                index = np.random.randint(0, len(dataset))
                states, actions, dones, goal, succ, extra_monitors, dense_goals, dense_dones, raw_data = dataset[index]

                # for i, action in enumerate(actions):
                #     state = states[i]
                #     domain.forward_features_and_axioms(state)

                #     _, next_predicted = action(state)
                #     domain.forward_derived_features(next_predicted)
                #     next_state = states[i+1]
                #     domain.forward_features_and_axioms(next_state)

                #     print(state)
                #     print('- action: {} ->'.format(action))
                #     print(next_predicted)
                #     print(next_state)

                #     images_1 = [i.reshape(3, 32, 32) for i in state.features['item-image'].tensor] + [i.reshape(3, 32, 32) for i in state.features['container-image'].tensor]
                #     images_2 = [i.reshape(3, 32, 32) for i in next_state.features['item-image'].tensor] + [i.reshape(3, 32, 32) for i in next_state.features['container-image'].tensor]

                #     ts.show([images_1, images_2])

                # batch_states = pds.BatchState.from_states(domain, states)
                # domain.forward_features_and_axioms(batch_states)
                # print(batch_states)
                # for subgoal in dense_goals:
                #     pred = domain.forward_expr(batch_states, [], subgoal).tensor
                #     pred = (pred * 100).round()
                #     print(subgoal, pred)

                # images = list()
                # for i, action in enumerate(actions + [None]):
                #     state = states[i]
                #     images_1 = [i.reshape(3, 32, 32) for i in state.features['item-image'].tensor] + [i.reshape(3, 32, 32) for i in state.features['container-image'].tensor]
                #     images.append(images_1)
                # ts.show(images)

                state = states[0]
                continuous_values = generate_continuous_values(domain, state)
                plan = pds.brute_force_search(
                    domain,
                    state,
                    dense_goals[3],
                    max_depth=3,
                    continuous_values=continuous_values,
                    use_tuple_desc=False,
                    use_quantized_state=False,
                    forward_augmented=True,
                    verbose=False,
                )
                print('actions:', plan)
                print('')
    else:
        ckpt_name = f'dumps/{args.baseline}-{args.task}-load={load_id}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

    # from IPython import embed; embed()


def generate_continuous_values(domain, state):
    poses = list()
    container_pose_tensor = state.features['container-pose'].tensor
    for i in range(len(container_pose_tensor)):
        pose = container_pose_tensor[i]
        poses.append(pds.Value(domain.types['pose'], [], pose + torch.tensor([0, 0, 0.01])))
        # poses.append(pose + torch.tensor([0, 0, 0.41]))
        # poses.append(pose + torch.tensor([0, -0.05, 0.01]))
        # poses.append(pose + torch.tensor([0, 0.05, 0.01]))
    return {'pose': poses}


if __name__ == '__main__':
    main()
