from .events import Features, Interaction
from itertools import count
import graph_tool.all as gt
import numpy as np


def gt_model(num_iters, num_users, activity_potential_generator, p_deletion,
             p_triadic_closure, link_reinforecement_inc=1.0,
             memory_strength=1.0, max_peer_influence_prob=0.0,
             peer_influence_thres=0.10, beta=1,
             log_features=False, tqdm=False):
    iterations = range
    if tqdm:
        from tqdm import tnrange
        iterations = tnrange

    graph = gt.Graph(directed=False)
    user_id_generator = count()

    event_log = []  # keeps track of all interactions between nodes
    features_log = []  # keeps track of some statistics in each iteration

    graph.graph_properties['p_triadic_closure'] = graph.new_gp('double')
    graph.graph_properties['link_reinforecement'] = graph.new_gp('double')
    graph.vertex_properties['last_t_active'] = graph.new_vp('long')
    graph.vertex_properties['activity_potential'] = graph.new_vp('double')
    graph.vertex_properties['uid'] = graph.new_vp('int')
    graph.edge_properties['weight'] = graph.new_ep('double')
    graph.graph_properties['p_triadic_closure'] = p_triadic_closure
    graph.graph_properties['link_reinforecement'] = link_reinforecement_inc

    avg_weight = 0

    for _ in range(num_users):
        node = graph.add_vertex()
        graph.vp.last_t_active[node] = -1
        graph.vp.uid[node] = next(user_id_generator)
        graph.vp.activity_potential[node] = next(activity_potential_generator)

    for t in iterations(num_iters):
        deletion_list = set()
        num_edges_before_iter = graph.num_edges()
        num_interactions_in_iter = 0

        nodes = list(graph.vertices())
        np.random.shuffle(nodes)
        for node in nodes:
            if np.random.rand() < p_deletion:
                deletion_list.add(node)
                continue

            is_intrinsic_active = (np.random.rand() <
                                   graph.vp.activity_potential[node])

            if beta == 'equal':
                rescale_weights_beta = 0
            elif beta == 'avg_weight':
                rescale_weights_beta = 1 / avg_weight if avg_weight > 0 else 1
            else:
                rescale_weights_beta = 1

            is_peer_influence_active = gt_peer_influence(
                    graph, node, t, max_peer_influence_prob, peer_influence_thres,
                    beta=rescale_weights_beta)

            if not is_intrinsic_active and not is_peer_influence_active:
                continue  # user is inactive

            num_interactions_in_iter += 1
            p_new_tie = memory_strength / (memory_strength + node.out_degree())

            if np.random.rand() < p_new_tie:
                if node.out_degree() == 0:
                    other_node = gt_focal_closure(graph, node)
                else:
                    other_node = gt_triadic_closure(graph, node)
            else:
                other_node = gt_get_random_neighbour(graph, node)
                edge = graph.edge(node, other_node)
                graph.ep.weight[edge] += graph.gp.link_reinforecement

            graph.vp.last_t_active[node] = t
            graph.vp.last_t_active[other_node] = t
            event_log.append(Interaction(
                t, graph.vp.uid[node], graph.vp.uid[other_node],
                not is_intrinsic_active and is_peer_influence_active)
            )

        try:
            avg_weight, _ = gt.edge_average(graph, graph.ep.weight)
        except ZeroDivisionError:
            avg_weight = 0  # in case that there wasn't an interaction yet

        if log_features:
            clustering = gt.local_clustering(graph)
            avg_cc, _ = gt.vertex_average(graph, clustering)
            avg_degree, _ = gt.vertex_average(graph, 'out')
            n_create = graph.num_edges() - num_edges_before_iter
            n_reinforce = num_interactions_in_iter - n_create
            features_log.append(
                Features(t, avg_degree, avg_cc, avg_weight, n_create,
                         n_reinforce)
            )

        graph.remove_vertex(deletion_list)
        for _ in range(len(deletion_list)):
            node = graph.add_vertex()
            graph.vp.last_t_active[node] = -1
            graph.vp.uid[node] = next(user_id_generator)
            graph.vp.activity_potential[node] = next(
                activity_potential_generator)

    return graph, event_log, features_log


def gt_focal_closure(graph, active_node, exceptions=None):
    if exceptions is None:
        exceptions = set()
    exceptions |= {active_node}
    exceptions |= set(active_node.out_neighbours())

    # it should be more efficient for large graphs to potentially redraw a node
    # index instead of explicity building a list of nodes that excludes the
    # exceptions and draw from it
    while True:
        node_idx = np.random.randint(graph.num_vertices())
        other_node = graph.vertex(node_idx)
        if other_node not in exceptions:
            break

    edge = graph.add_edge(active_node, other_node)
    graph.ep.weight[edge] = 1.0
    return other_node


def gt_triadic_closure(graph, active_node):
    neighbour = gt_get_random_neighbour(graph, active_node)
    if neighbour.out_degree() == 1:
        return gt_focal_closure(graph, active_node)

    other_node = gt_get_random_neighbour(graph, neighbour,
                                         exceptions={active_node})
    if other_node in active_node.out_neighbours():
        edge = graph.edge(active_node, other_node)
        graph.ep.weight[edge] += graph.gp.link_reinforecement
        return other_node

    if np.random.rand() < graph.gp.p_triadic_closure:
        edge = graph.add_edge(active_node, other_node)
        graph.ep.weight[edge] = 1.0
        return other_node

    return gt_focal_closure(graph, active_node, exceptions={other_node})


def gt_get_random_neighbour(graph, node, exceptions=None):
    if exceptions is None:
        exceptions = set()

    neighbours = [neighbour for neighbour in node.out_neighbours()
                  if neighbour not in exceptions]
    total_weight = sum(graph.ep.weight[graph.edge(node, neighbour)]
                       for neighbour in neighbours)
    weight_dist = [graph.ep.weight[graph.edge(node, neighbour)] / total_weight
                   for neighbour in neighbours]
    return np.random.choice(neighbours, p=weight_dist)


def gt_peer_influence(graph, node, t, max_prob, critical_thres, beta):
    rand_value = np.random.rand()

    if (max_prob == 0 or rand_value >= max_prob or node.out_degree() == 0 or not
       any(graph.vp.last_t_active[n] == t-1 for n in node.out_neighbours())):
        return False

    weights_last_active = total_weight = 0
    for neighbour in node.out_neighbours():
        weight = graph.ep.weight[graph.edge(node, neighbour)]
        weight = np.exp(weight * beta)  # rescale weight (softmax)
        if graph.vp.last_t_active[neighbour] == t-1:
            weights_last_active += weight
        total_weight += weight

    # check for overflows
    if np.isinf(weights_last_active):
        alpha = 1
    elif np.isinf(total_weight):
        alpha = 0
    else:
        alpha = weights_last_active / total_weight
    
    peer_influence_prob = (max_prob * alpha) / np.sqrt(critical_thres**2 + alpha**2)
    return rand_value < peer_influence_prob
