from .events import Interaction
from itertools import count
import numpy as np
import warnings


class User(object):
    def __init__(self, user_id, activity_potential, tie_inc):
        self.user_id = user_id
        self.activity_potential = activity_potential
        self.neighborhood = Neighborhood(tie_inc)
        self.last_active = -1


class Neighborhood(object):
    def __init__(self, tie_inc):
        self.tie_inc = tie_inc
        self.weights = dict()

    def new_tie(self, user):
        assert user not in self.weights
        self.weights[user] = 1.0

    def reinforce_tie(self, user):
        assert user in self.weights
        self.weights[user] += self.tie_inc

    def delete_tie(self, user):
        assert user in self.weights
        del self.weights[user]

    def get_random_neighbor_id(self, exceptions=None):
        if exceptions is None:
            exceptions = set()

        users = []
        weights = []
        for user_id, weight in self.weights.items():
            if user_id in exceptions:
                continue
            users.append(user_id)
            weights.append(weight)

        total_weight = np.sum(weights)
        return np.random.choice(
            users, p=[weight / total_weight for weight in weights]
        )

    def get_neighbor_ids(self):
        return list(self.weights.keys())

    def __len__(self):
        return len(self.weights)


def model(num_iters, num_users, activity_potential_generator, p_deletion,
          p_triadic_closure, link_reinforecement_inc=1.0,
          memory_strength=1.0, max_peer_influence_prob=0.0,
          peer_influence_thes=0.1, beta=1):
    interactions = []  # keeps track of all interactions between users
    deleted_users = []

    avg_weight = 0

    users = dict()
    user_id_generator = count()
    for _ in range(num_users):
        user_id = next(user_id_generator)
        activity = next(activity_potential_generator)
        users[user_id] = User(user_id, activity, link_reinforecement_inc)

    for t in range(num_iters):
        user_ids = list(users.keys())
        np.random.shuffle(user_ids)

        for user_id in user_ids:
            user = users[user_id]

            if np.random.rand() < p_deletion:
                deleted_users.append(user_id)
                delete_user(users, user_id)
                new_user_id = next(user_id_generator)
                users[new_user_id] = User(new_user_id,
                                          next(activity_potential_generator),
                                          link_reinforecement_inc)
                continue

            is_intrinsic_active = np.random.rand() < user.activity_potential

            if beta == 'equal':
                rescale_weights_beta = 0
            elif beta == 'avg_weight':
                rescale_weights_beta = 1 / avg_weight if avg_weight > 0 else 1
            else:
                rescale_weights_beta = 1

            is_peer_influence_active = peer_influence(
                users, user, t, max_peer_influence_prob, peer_influence_thes,
                beta=rescale_weights_beta)

            if not is_intrinsic_active and not is_peer_influence_active:
                continue  # user is inactive

            user_degree = len(user.neighborhood)
            p_new_tie = memory_strength / (memory_strength + user_degree)

            if np.random.rand() < p_new_tie:
                if user_degree == 0:
                    other_user = focal_closure(users, user_id)
                else:
                    other_user = triadic_closure(users, user_id,
                                                 p_triadic_closure)
            else:
                other_user_id = user.neighborhood.get_random_neighbor_id()
                other_user = users[other_user_id]
                user.neighborhood.reinforce_tie(other_user_id)
                other_user.neighborhood.reinforce_tie(user_id)

            user.last_active = other_user.last_active = t
            interactions.append(Interaction(
                t, user_id, other_user.user_id,
                not is_intrinsic_active and is_peer_influence_active)
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_weight = np.nanmean([
                np.nanmean(list(user.neighborhood.weights.values())) / 2
                for _, user in users.items()
            ])

    return interactions, deleted_users


def focal_closure(users, active_user_id, exceptions=None):
    active_user = users[active_user_id]

    if exceptions is None:
        exceptions = set()
    exceptions |= {active_user_id, *active_user.neighborhood.get_neighbor_ids()}
    user_ids = set(users.keys()) - exceptions

    other_user_id = np.random.choice(list(user_ids))
    other_user = users[other_user_id]
    active_user.neighborhood.new_tie(other_user_id)
    other_user.neighborhood.new_tie(active_user_id)
    return other_user


def triadic_closure(users, active_user_id, p_triadic_closure):
    active_user = users[active_user_id]

    neighbor_id = active_user.neighborhood.get_random_neighbor_id()
    neighbor = users[neighbor_id]
    if len(neighbor.neighborhood) == 1:
        assert not neighbor.neighborhood.weights.get(active_user_id) is None
        return focal_closure(users, active_user_id)

    other_user_id = neighbor.neighborhood.get_random_neighbor_id(
        exceptions={active_user_id})
    other_user = users[other_user_id]

    if other_user_id in active_user.neighborhood.get_neighbor_ids():
        active_user.neighborhood.reinforce_tie(other_user_id)
        other_user.neighborhood.reinforce_tie(active_user_id)
        return other_user

    if np.random.rand() < p_triadic_closure:
        active_user.neighborhood.new_tie(other_user_id)
        other_user.neighborhood.new_tie(active_user_id)
        return other_user

    return focal_closure(users, active_user_id, exceptions={other_user_id})


def delete_user(users, active_user_id):
    active_user = users[active_user_id]
    for neighbor_id in active_user.neighborhood.get_neighbor_ids():
        users[neighbor_id].neighborhood.delete_tie(active_user_id)
    del users[active_user_id]


def peer_influence(users, active_user, t, max_prob, critical_thres, beta):
    rand_value = np.random.rand()

    if max_prob == 0 or rand_value > max_prob or not len(active_user.neighborhood):
        return False

    weights_last_active = total_weight = 0
    for neighbor_id, weight in active_user.neighborhood.weights.items():
        scaled_weight = np.exp(weight * beta)
        if users[neighbor_id].last_active == t-1:
            weights_last_active += scaled_weight
        total_weight += scaled_weight

    # check for overflows
    if np.isinf(weights_last_active):
        alpha = 1
    elif np.isinf(total_weight):
        alpha = 0
    else:
        alpha = weights_last_active / total_weight

    peer_influence_prob = (max_prob * alpha) / np.sqrt(critical_thres**2 + alpha**2)
    return rand_value < peer_influence_prob
