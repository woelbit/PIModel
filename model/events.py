from collections import namedtuple


Interaction = namedtuple('Interaction', [
    'time', 'active_user', 'other_user', 'peer_influenced'])
Features = namedtuple('Features', [
    'time', 'avg_degree', 'avg_clustering_coeff', 'avg_weight', 'n_create',
    'n_reinforce'])
