#!/usr/bin/env python3
import os
import sys
import re
import argparse

import numpy as np
import tensorflow as tf
import networkx as nx

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# This script lives ABOVE both repos
TOP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ERLANG_DIR = os.path.join(TOP_BASE_DIR, "RouteNet-Erlang")
FERMI_DIR = os.path.join(TOP_BASE_DIR, "RouteNet-Fermi")

ERLANG_SCHED_DIR = os.path.join(ERLANG_DIR, "Scheduling")
FERMI_SCHED_DIR = os.path.join(FERMI_DIR, "scheduling")


# =============================================================================
# Shared helpers
# =============================================================================

def parse_candidate_routes(routes_file):
    """
    Expect lines like:
        1 (hops=4): 0->2->4->8->5
        2 (hops=6): 0->2->4->8->7->6->5

    Returns: [{'id': '1', 'src': 0, 'dst': 5, 'nodes': [0,2,4,8,5]}, ...]
    """
    routes = []
    with open(routes_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            left = left.strip()
            right = right.strip()

            cand_id = left.split()[0]

            node_strs = [s.strip() for s in right.split("->")]
            nodes = [int(s) for s in node_strs if s != ""]
            if len(nodes) < 2:
                continue
            src = nodes[0]
            dst = nodes[-1]
            routes.append({"id": cand_id, "src": src, "dst": dst, "nodes": nodes})
    return routes


def validate_candidate_path_for_sample(sample, cand_id, path_nodes, framework_tag, metric):
    """
    Checks that every hop (u,v) in path_nodes is a valid directed edge
    in this sample's topology. If not, exit with a clear message.
    """
    G_topo = sample.get_topology_object()
    for u, v in zip(path_nodes, path_nodes[1:]):
        if not G_topo.has_edge(u, v):
            raise SystemExit(
                f"[{framework_tag}] Candidate {cand_id} has invalid hop {u}->{v} "
                f"for this topology.\n"
                f"  metric        = {metric}\n"
                f"Make sure the routes_file was generated for THIS topology/sample."
            )


def create_datanet_iterator(DatanetAPI_cls, dataset_dir, shuffle):
    """
    Helper that works with both 2-arg and 3-arg DatanetAPI constructors.
    """
    try:
        tool = DatanetAPI_cls(dataset_dir, [], shuffle)
    except TypeError:
        tool = DatanetAPI_cls(dataset_dir, shuffle)
    return iter(tool)


# =============================================================================
# ERLANG SCHEDULING BRANCH
# =============================================================================

def erlang_sched_import_modules():
    """
    Add RouteNet-Erlang/Scheduling to sys.path and import scheduling modules.
    """
    if ERLANG_SCHED_DIR not in sys.path:
        sys.path.insert(0, ERLANG_SCHED_DIR)

    from datanetAPI import DatanetAPI
    from model import GNN_Model
    from read_dataset import sample_to_dependency_graph

    return DatanetAPI, GNN_Model, sample_to_dependency_graph


def erlang_sched_build_graph_and_features(sample, label, sample_to_dependency_graph_fn):
    """
    Build features for a single scheduling sample, matching Scheduling/read_dataset.py.

    Returns:
      x: dict of numpy arrays
      y: numpy array of labels (in ORIGINAL scale)
      path_src_dst: list[(src,dst)]
      path_routes: list[str]
    """
    # Dependency graph + basic counts from original helper
    D_G, n_q, n_p, n_l = sample_to_dependency_graph_fn(sample)

    # --- adjacency structures (copied from Scheduling/read_dataset.generator) ---
    link_to_path = np.array([], dtype="int32")
    queue_to_path = np.array([], dtype="int32")
    l_p_s = np.array([], dtype="int32")
    l_q_p = np.array([], dtype="int32")
    path_ids = np.array([], dtype="int32")

    for i in range(n_p):
        l_s_l = 0
        q_s_l = 0
        for elem in D_G[f"p_{i}"]:
            if elem.startswith("l_"):
                link_to_path = np.append(link_to_path, int(elem.replace("l_", "")))
                l_s_l += 1
            elif elem.startswith("q_"):
                queue_to_path = np.append(queue_to_path, int(elem.replace("q_", "")))
                q_s_l += 1
        path_ids = np.append(path_ids, [i] * q_s_l)
        l_p_s = np.append(l_p_s, range(l_s_l))
        l_q_p = np.append(l_q_p, range(q_s_l))

    path_to_queue = np.array([], dtype="int32")
    sequence_queues = np.array([], dtype="int32")
    for i in range(n_q):
        seq_len = 0
        for elem in D_G[f"q_{i}"]:
            path_to_queue = np.append(path_to_queue, int(elem.replace("p_", "")))
            seq_len += 1
        sequence_queues = np.append(sequence_queues, [i] * seq_len)

    queue_to_link = np.array([], dtype="int32")
    sequence_links = np.array([], dtype="int32")
    l_q_l = np.array([], dtype="int32")
    for i in range(n_l):
        seq_len = 0
        for elem in D_G[f"l_{i}"]:
            queue_to_link = np.append(queue_to_link, int(elem.replace("q_", "")))
            seq_len += 1
        sequence_links = np.append(sequence_links, [i] * seq_len)
        l_q_l = np.append(l_q_l, range(seq_len))

    # Skip logic: if any path has 0 jitter or 0 delay, sample is skipped
    if 0 in list(nx.get_node_attributes(D_G, "jitter").values()) or \
       0 in list(nx.get_node_attributes(D_G, "delay").values()):
        raise ValueError("Scheduling sample has zero jitter or delay and is skipped in training.")

    # Features as in generator
    x = {
        "traffic": np.array(list(nx.get_node_attributes(D_G, "traffic").values()), dtype=np.float32),
        "packets": np.array(list(nx.get_node_attributes(D_G, "packets").values()), dtype=np.float32),
        "capacity": np.array(list(nx.get_node_attributes(D_G, "capacity").values()), dtype=np.float32),
        "policy": np.array(list(nx.get_node_attributes(D_G, "policy").values()), dtype=np.int32),
        "priority": np.array(list(nx.get_node_attributes(D_G, "priority").values()), dtype=np.int32),
        "weight": np.array(
            [w / 100.0 for w in list(nx.get_node_attributes(D_G, "weight").values())],
            dtype=np.float32,
        ),
        "link_to_path": link_to_path.astype(np.int32),
        "queue_to_path": queue_to_path.astype(np.int32),
        "path_to_queue": path_to_queue.astype(np.int32),
        "queue_to_link": queue_to_link.astype(np.int32),
        "sequence_queues": sequence_queues.astype(np.int32),
        "sequence_links": sequence_links.astype(np.int32),
        "path_ids": path_ids.astype(np.int32),
        "l_p_s": l_p_s.astype(np.int32),
        "l_q_p": l_q_p.astype(np.int32),
        "l_q_l": l_q_l.astype(np.int32),
        "n_queues": np.array(n_q, dtype=np.int32),
        "n_links": np.array(n_l, dtype=np.int32),
        "n_paths": np.array(n_p, dtype=np.int32),
    }

    if label not in ("delay", "jitter", "drops"):
        raise ValueError(f"Unsupported scheduling label '{label}' (expected 'delay', 'jitter' or 'drops').")

    # ORIGINAL-SCALE labels
    y = np.array(list(nx.get_node_attributes(D_G, label).values()), dtype=np.float32)

    # Per-path src/dst + route string using routing matrix
    R = sample.get_routing_matrix()
    path_src_dst = []
    path_routes = []
    for i in range(n_p):
        src = D_G.nodes[f"p_{i}"]["source"]
        dst = D_G.nodes[f"p_{i}"]["destination"]
        path_src_dst.append((src, dst))

        path_nodes = R[src, dst]
        if path_nodes is None or len(path_nodes) == 0:
            route_str = "(no path)"
        else:
            route_str = "->".join(str(n) for n in path_nodes)
        path_routes.append(route_str)

    return x, y, path_src_dst, path_routes, D_G


def erlang_sched_get_sample_features(dataset_dir, label, sample_index,
                                     DatanetAPI_cls, sample_to_dependency_graph_fn,
                                     graph_filter=None):
    """
    Iterate over DatanetAPI and return the sample_index-th *valid* scheduling sample.
    Valid means:
      - passes the zero-delay/jitter skip logic, AND
      - (if graph_filter is not None) its graph file name contains graph_filter.
    """
    it = create_datanet_iterator(DatanetAPI_cls, dataset_dir, shuffle=False)
    valid_idx = -1

    while True:
        try:
            sample = next(it)
        except StopIteration:
            if graph_filter is None:
                raise RuntimeError(
                    f"[Erlang-Scheduling] Reached end of dataset before index {sample_index}. "
                    f"Found {valid_idx + 1} valid samples."
                )
            else:
                raise RuntimeError(
                    f"[Erlang-Scheduling] Reached end of dataset before index {sample_index} "
                    f"when filtering by graph '{graph_filter}'. "
                    f"Found {valid_idx + 1} valid samples matching that filter."
                )

        # Optional graph filter by graph file name (e.g. 'geant2', 'nsfnet', 'rediris')
        if graph_filter is not None:
            graph_name = getattr(sample, "_graph_file", None)
            if graph_name is None or graph_filter not in graph_name:
                continue  # skip this sample entirely

        try:
            x, y, path_src_dst, path_routes, _ = erlang_sched_build_graph_and_features(
                sample, label, sample_to_dependency_graph_fn
            )
        except ValueError:
            # e.g. zero jitter/delay sample, skip it
            continue

        valid_idx += 1
        if valid_idx == sample_index:
            graph_name = getattr(sample, "_graph_file", "unknown")
            dataset_file = sample._get_data_set_file_name()
            print(f"[Erlang-Sched] Selected sample_index={sample_index} "
                  f"(graph={graph_name}, dataset_file={dataset_file})")
            return sample, x, y, path_src_dst, path_routes



def erlang_sched_build_features_with_override(sample, label, override_dict,
                                              sample_to_dependency_graph_fn):
    """
    Override routing for some (src,dst) pairs in a scheduling sample,
    rebuild features, then restore original routing.
    """
    R_orig = sample.get_routing_matrix().copy()
    R_mod = R_orig.copy()
    for (s, d), path in override_dict.items():
        R_mod[s, d] = path
    sample._set_routing_matrix(R_mod)

    try:
        x, y, path_src_dst, path_routes, _ = erlang_sched_build_graph_and_features(
            sample, label, sample_to_dependency_graph_fn
        )
    finally:
        sample._set_routing_matrix(R_orig)

    return x, y, path_src_dst, path_routes


def erlang_sched_transformation(x, y, metric):
    """
    Apply the same normalization / log transform as the scheduling scripts.

    For delay & jitter: same stats, log(y)
    For loss (drops): different stats, no log.
    """
    metric = metric.lower()
    if metric in ("delay", "jitter"):
        traffic_mean = 660.5723876953125
        traffic_std = 420.22003173828125
        packets_mean = 0.6605737209320068
        packets_std = 0.42021000385284424
        capacity_mean = 25442.669921875
        capacity_std = 16217.9072265625

        x["traffic"] = (x["traffic"] - traffic_mean) / traffic_std
        x["packets"] = (x["packets"] - packets_mean) / packets_std
        x["capacity"] = (x["capacity"] - capacity_mean) / capacity_std

        return x, tf.math.log(y)
    else:  # loss / drops
        traffic_mean = 1650.59814453125
        traffic_std = 855.7061767578125
        packets_mean = 1.650602102279663
        packets_std = 0.8556720614433289
        capacity_mean = 25457.9453125
        capacity_std = 16221.1337890625

        x["traffic"] = (x["traffic"] - traffic_mean) / traffic_std
        x["packets"] = (x["packets"] - packets_mean) / packets_std
        x["capacity"] = (x["capacity"] - capacity_mean) / capacity_std

        return x, y  # no log for losses


def erlang_sched_denorm_MAPE(y_true, y_pred):
    # Same as Delay/Jitter check_predictions.py
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100.0


def erlang_sched_load_best_model(params, ckpt_dir, metric, GNN_Model_cls):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(params["HYPERPARAMETERS"]["learning_rate"])
    )
    model = GNN_Model_cls(params)
    loss_object = tf.keras.losses.MeanSquaredError()

    model.compile(
        loss=loss_object,
        optimizer=optimizer,
        run_eagerly=False,
        metrics=erlang_sched_denorm_MAPE,  # keep whatever you already use
    )

    best = None
    best_mre = float("inf")

    for f in os.listdir(ckpt_dir):
        full_path = os.path.join(ckpt_dir, f)
        if os.path.isfile(full_path):
            reg = re.findall(r"\d+\.\d+", f)
            if reg:
                mre = float(reg[0])
                if mre <= best_mre:
                    # Start from filename without ".index"
                    best_candidate = f.replace(".index", "")
                    # If it's a ".data-00000-of-00001" file, strip that suffix
                    if ".data" in best_candidate:
                        idx = best_candidate.rfind(".")
                        best_candidate = best_candidate[:idx]
                    best = best_candidate
                    best_mre = mre

    if best is None:
        raise RuntimeError(f"[Erlang-Sched] No checkpoint file found in {ckpt_dir}")

    print(f"BEST CHECKPOINT FOUND (Erlang-Sched): {best}")
    model.load_weights(os.path.join(ckpt_dir, best))
    return model



def run_erlang_scheduling(args):
    import configparser

    metric = args.metric.lower()
    if metric not in ("delay", "jitter", "loss"):
        raise SystemExit("Erlang scheduling supports metrics: delay, jitter, loss.")

    label_map = {"delay": "delay", "jitter": "jitter", "loss": "drops"}
    label = label_map[metric]

    DatanetAPI, GNN_Model, sample_to_dependency_graph_fn = erlang_sched_import_modules()

    dataset_dir = os.path.join(ERLANG_DIR, "data", "scheduling", args.dataset_split)

    # Metric-specific experiment dirs
    if metric == "delay":
        exp_dir = os.path.join(ERLANG_SCHED_DIR, "Delay")
        ckpt_dir = os.path.join(exp_dir, "ckpt_dir_reg")
    elif metric == "jitter":
        exp_dir = os.path.join(ERLANG_SCHED_DIR, "Jitter")
        ckpt_dir = os.path.join(exp_dir, "ckpt_dir")
    else:  # loss
        exp_dir = os.path.join(ERLANG_SCHED_DIR, "Losses")
        ckpt_dir = os.path.join(exp_dir, "ckpt_dir")


    print(f"\n=== ERLANG SCHEDULING MODE B ===")
    print(f"Dataset directory : {dataset_dir}")
    print(f"Experiment dir    : {exp_dir}")
    print(f"Metric / label    : {metric} (D_G attribute = '{label}')")
    print(f"Sample index      : {args.sample_index}")
    print(f"Routes file       : {args.routes_file}\n")

    params = configparser.ConfigParser()
    params._interpolation = configparser.ExtendedInterpolation()
    params.read(os.path.join(exp_dir, "config.ini"))

    sample, x_base, y_base, path_src_dst_base, path_routes_base = erlang_sched_get_sample_features(
        dataset_dir,
        label,
        args.sample_index,
        DatanetAPI,
        sample_to_dependency_graph_fn,
        graph_filter=args.graph,
    )

    graph_file = getattr(sample, "_graph_file", None)
    print(f"Baseline sample (Erlang-Sched) uses topology graph: {graph_file}")

    n_paths = int(x_base["n_paths"])
    print(f"Loaded baseline sample (Erlang-Sched): n_paths = {n_paths}, n_links = {int(x_base['n_links'])}")

    model = erlang_sched_load_best_model(params, ckpt_dir, metric, GNN_Model)

    candidates = parse_candidate_routes(args.routes_file)
    if not candidates:
        raise RuntimeError(f"No candidate routes parsed from {args.routes_file}")

    # Check that all candidates share the same (src, dst)
    first_src = candidates[0]["src"]
    first_dst = candidates[0]["dst"]
    for c in candidates:
        if c["src"] != first_src or c["dst"] != first_dst:
            print("WARNING: Not all candidates share the same (src,dst); interpretation is trickier.")
            break

    np.set_printoptions(precision=6, suppress=True, linewidth=160)

    label_print = "loss" if metric == "loss" else metric

    for cand in candidates:
        cand_id = cand["id"]
        c_src = cand["src"]
        c_dst = cand["dst"]
        path_nodes = cand["nodes"]

        # Validate candidate path against this topology
        validate_candidate_path_for_sample(
            sample, cand_id, path_nodes, framework_tag="Erlang-Sched", metric=metric
        )

        print("\n" + "=" * 120)
        print(f"[Erlang-Sched] Candidate {cand_id}: src={c_src}, dst={c_dst}, path = {'->'.join(map(str, path_nodes))}")
        print("=" * 120)

        override = {(c_src, c_dst): path_nodes}
        x_cand, _, path_src_dst_cand, path_routes_cand = erlang_sched_build_features_with_override(
            sample, label, override, sample_to_dependency_graph_fn
        )

        if path_src_dst_cand != path_src_dst_base:
            raise RuntimeError("Path ordering mismatch between baseline and candidate features (Erlang-Sched).")

        # Convert to tensors + normalization
        x_tf = {k: tf.convert_to_tensor(v) for k, v in x_cand.items()}
        y_tf = tf.convert_to_tensor(y_base)
        x_tf_norm, _ = erlang_sched_transformation(x_tf, y_tf, metric)

        preds_raw = model(x_tf_norm, training=False).numpy().reshape(-1)

        # For delay/jitter, labels are log-transformed, so exponentiate predictions
        if metric in ("delay", "jitter"):
            preds = np.exp(preds_raw)
        else:
            preds = preds_raw

        if preds.shape[0] != n_paths:
            raise RuntimeError("Predictions length mismatch with n_paths (Erlang-Sched).")

        header = (
           "  idx src dst            route_nodes    traffic    packets   "
           f"true_{label_print}_orig   pred_{label_print}"
        )

        print(header)
        print("-" * len(header))

        traffic_arr = np.array(x_base["traffic"]).reshape(-1)
        packets_arr = np.array(x_base["packets"]).reshape(-1)

        for i in range(n_paths):
            src_i, dst_i = path_src_dst_cand[i]
            route_str = path_routes_cand[i]

            traffic_i = float(traffic_arr[i])
            packets_i = float(packets_arr[i])
            true_val = float(y_base[i])
            pred_val = float(preds[i])

            mark = "*" if (src_i == c_src and dst_i == c_dst) else " "

            print(
                f"{i:5d} {src_i:3d} {dst_i:3d} {mark}{route_str:>25s} "
                f"{traffic_i:10.3f} {packets_i:10.3f} "
                f"{true_val:17.6f} {pred_val:17.6f}"
            )


# =============================================================================
# FERMI SCHEDULING BRANCH
# =============================================================================

def fermi_sched_import_modules(metric):
    """
    Add RouteNet-Fermi/scheduling to sys.path and import metric-specific
    model + data_generator for scheduling.
    """
    if FERMI_SCHED_DIR not in sys.path:
        sys.path.insert(0, FERMI_SCHED_DIR)
    if FERMI_DIR not in sys.path:
        sys.path.insert(0, FERMI_DIR)

    from datanetAPI import DatanetAPI

    metric = metric.lower()
    if metric == "delay":
        from delay_model import RouteNet_Fermi as FermiModel
        from delay.data_generator import network_to_hypergraph, hypergraph_to_input_data
        subdir = "delay"
        ckpt_dir_name = "ckpt_dir"
    elif metric == "jitter":
        from jitter_model import RouteNet_Fermi as FermiModel
        from jitter.data_generator import network_to_hypergraph, hypergraph_to_input_data
        subdir = "jitter"
        ckpt_dir_name = "ckpt_dir"
    elif metric == "loss":
        from loss_model import RouteNet_Fermi as FermiModel
        from losses.data_generator import network_to_hypergraph, hypergraph_to_input_data
        subdir = "losses"
        ckpt_dir_name = "ckpt_dir"
    else:
        raise SystemExit("Unsupported Fermi scheduling metric (should be delay, jitter or loss).")

    return DatanetAPI, FermiModel, network_to_hypergraph, hypergraph_to_input_data, subdir, ckpt_dir_name


def fermi_sched_build_graph_and_features(sample, network_to_hypergraph, hypergraph_to_input_data):
    """
    Build RouteNet-Fermi scheduling hypergraph + inputs for a single sample.
    """
    G = nx.DiGraph(sample.get_topology_object())
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()
    R = sample.get_routing_matrix()

    HG = network_to_hypergraph(G=G, R=R, T=T, P=P)

    x_dict, y_list = hypergraph_to_input_data(HG)

    traffic_attr = nx.get_node_attributes(HG, "traffic")
    p_nodes = list(traffic_attr.keys())

    path_src_dst = []
    path_routes = []
    for p_name in p_nodes:
        src = HG.nodes[p_name]["source"]
        dst = HG.nodes[p_name]["destination"]
        path_src_dst.append((src, dst))

        path_nodes = R[src, dst]
        if path_nodes is None or len(path_nodes) == 0:
            route_str = "(no path)"
        else:
            route_str = "->".join(str(n) for n in path_nodes)
        path_routes.append(route_str)

    y = np.array(y_list, dtype=np.float32)

    return x_dict, y, path_src_dst, path_routes


def fermi_sched_get_sample_features(dataset_dir, sample_index, metric,
                                    DatanetAPI_cls, network_to_hypergraph, hypergraph_to_input_data,
                                    graph_filter=None):
    """
    Iterate over DatanetAPI and return the sample_index-th scheduling sample
    after applying an optional graph_filter (substring on sample._graph_file).

    No extra label-based skipping here; we keep all samples.
    """
    it = create_datanet_iterator(DatanetAPI_cls, dataset_dir, shuffle=False)

    raw_idx = -1       # index over all samples from DatanetAPI
    filtered_idx = -1  # index over samples passing graph_filter (if any)

    for raw_idx, sample in enumerate(it):
        # Optional graph filter
        if graph_filter is not None:
            graph_name = getattr(sample, "_graph_file", None)
            if graph_name is None or graph_filter not in graph_name:
                continue

        filtered_idx += 1
        if filtered_idx < sample_index:
            continue

        x_dict, y, path_src_dst, path_routes = fermi_sched_build_graph_and_features(
            sample, network_to_hypergraph, hypergraph_to_input_data
        )

        graph_name = getattr(sample, "_graph_file", "unknown")
        dataset_file = sample._get_data_set_file_name()
        print(f"[Fermi-Sched] Selected sample_index={sample_index} "
              f"(graph={graph_name}, dataset_file={dataset_file})")

        return sample, x_dict, y, path_src_dst, path_routes

    # If we exit the loop, the iterator is exhausted.
    if graph_filter is None:
        # No filter case
        if raw_idx == -1:
            raise RuntimeError(
                f"[Fermi-Scheduling] Dataset '{dataset_dir}' appears to be empty or not readable "
                f"(no samples yielded by DatanetAPI)."
            )
        else:
            raise RuntimeError(
                f"[Fermi-Scheduling] Dataset '{dataset_dir}' has only {raw_idx + 1} samples; "
                f"cannot access sample_index={sample_index}."
            )
    else:
        if filtered_idx == -1:
            raise RuntimeError(
                f"[Fermi-Scheduling] No samples found matching graph filter '{graph_filter}' "
                f"in dataset '{dataset_dir}'."
            )
        else:
            raise RuntimeError(
                f"[Fermi-Scheduling] Dataset '{dataset_dir}' has only {filtered_idx + 1} samples "
                f"matching graph filter '{graph_filter}'; cannot access sample_index={sample_index}."
            )



def fermi_sched_build_features_with_override(sample, override_dict,
                                             network_to_hypergraph, hypergraph_to_input_data):
    """
    Override routing for some (src,dst) pairs and rebuild Fermi features.
    """
    R_orig = sample.get_routing_matrix().copy()
    R_mod = R_orig.copy()
    for (s, d), path in override_dict.items():
        R_mod[s, d] = path
    sample._set_routing_matrix(R_mod)

    try:
        x_dict, y, path_src_dst, path_routes = fermi_sched_build_graph_and_features(
            sample, network_to_hypergraph, hypergraph_to_input_data
        )
    finally:
        sample._set_routing_matrix(R_orig)

    return x_dict, path_src_dst, path_routes


def fermi_sched_to_tensor_dict(x_dict):
    """
    Convert a Fermi input dict (numpy / lists / ragged) into a dict of tensors
    with the exact dtypes expected by the RouteNet-Fermi models.
    """
    float_keys = {
        "traffic",
        "packets",
        "eq_lambda",
        "avg_pkts_lambda",
        "exp_max_factor",
        "pkts_lambda_on",
        "avg_t_off",
        "avg_t_on",
        "ar_a",
        "sigma",
        "capacity",
        "queue_size",
        "weight",
    }
    int_keys = {"length", "model", "policy", "priority"}

    out = {}
    for k, v in x_dict.items():
        # Ragged tensors already have correct dtypes
        if isinstance(v, tf.RaggedTensor):
            out[k] = v
            continue

        arr = np.array(v)
        if k in int_keys:
            out[k] = tf.convert_to_tensor(arr, dtype=tf.int32)
        elif k in float_keys:
            out[k] = tf.convert_to_tensor(arr, dtype=tf.float32)
        else:
            # Fallback: infer from numpy dtype
            if np.issubdtype(arr.dtype, np.integer):
                out[k] = tf.convert_to_tensor(arr, dtype=tf.int32)
            else:
                out[k] = tf.convert_to_tensor(arr, dtype=tf.float32)

    return out


def fermi_sched_load_best_model(ckpt_dir, FermiModel_cls, example_x):
    """
    Build and load RouteNet-Fermi model from scheduling ckpt_dir_* by choosing min-MRE ckpt.
    """
    model = FermiModel_cls()

    # Build variables by a forward pass using correct dtypes
    x_tf = fermi_sched_to_tensor_dict(example_x)
    _ = model(x_tf, training=False)

    best = None
    best_mre = float("inf")
    for f in os.listdir(ckpt_dir):
        full_path = os.path.join(ckpt_dir, f)
        if os.path.isfile(full_path):
            reg = re.findall(r"\d+\.\d+", f)
            if reg:
                mre = float(reg[0])
                if mre <= best_mre:
                    best = f.replace(".index", "")
                    if ".data" in best:
                        idx = best.rfind(".")
                        best = best[:idx]
                    best_mre = mre

    if best is None:
        raise RuntimeError(f"[Fermi-Scheduling] No checkpoint file found in {ckpt_dir}")

    print(f"BEST CHECKPOINT FOUND (Fermi-Sched): {best}")
    model.load_weights(os.path.join(ckpt_dir, best))
    return model


def run_fermi_scheduling(args):
    metric = args.metric.lower()
    if metric not in ("delay", "jitter", "loss"):
        raise SystemExit("Fermi scheduling supports metrics: delay, jitter, loss.")

    DatanetAPI, FermiModel, network_to_hypergraph, hypergraph_to_input_data, subdir, ckpt_dir_name = (
        fermi_sched_import_modules(metric)
    )

    dataset_dir = os.path.join(FERMI_DIR, "data", "scheduling", args.dataset_split)
    ckpt_dir = os.path.join(FERMI_SCHED_DIR, subdir, ckpt_dir_name)

    print(f"\n=== FERMI SCHEDULING MODE B (metric={metric}) ===")
    print(f"Dataset directory : {dataset_dir}")
    print(f"Checkpoint dir    : {ckpt_dir}")
    print(f"Sample index      : {args.sample_index}")
    print(f"Routes file       : {args.routes_file}\n")

    sample, x_base, y_base, path_src_dst_base, path_routes_base = fermi_sched_get_sample_features(
        dataset_dir,
        args.sample_index,
        metric,
        DatanetAPI,
        network_to_hypergraph,
        hypergraph_to_input_data,
        graph_filter=args.graph,
    )

    graph_file = getattr(sample, "_graph_file", None)
    print(f"Baseline sample (Fermi-Sched) uses topology graph: {graph_file}")


    n_paths = len(y_base)
    print(f"Loaded baseline sample (Fermi-Sched): n_paths = {n_paths}")

    model = fermi_sched_load_best_model(ckpt_dir, FermiModel, x_base)

    candidates = parse_candidate_routes(args.routes_file)
    if not candidates:
        raise RuntimeError(f"No candidate routes parsed from {args.routes_file}")

    first_src = candidates[0]["src"]
    first_dst = candidates[0]["dst"]
    for c in candidates:
        if c["src"] != first_src or c["dst"] != first_dst:
            print("WARNING: Not all candidates share the same (src,dst); interpretation is trickier.")
            break

    np.set_printoptions(precision=6, suppress=True, linewidth=160)

    for cand in candidates:
        cand_id = cand["id"]
        c_src = cand["src"]
        c_dst = cand["dst"]
        path_nodes = cand["nodes"]

        validate_candidate_path_for_sample(
            sample, cand_id, path_nodes, framework_tag="Fermi-Sched", metric=metric
        )

        print("\n" + "=" * 120)
        print(f"[Fermi-Sched] Candidate {cand_id}: src={c_src}, dst={c_dst}, path = {'->'.join(map(str, path_nodes))}")
        print("=" * 120)

        override = {(c_src, c_dst): path_nodes}
        x_cand, path_src_dst_cand, path_routes_cand = fermi_sched_build_features_with_override(
            sample,
            override,
            network_to_hypergraph,
            hypergraph_to_input_data,
        )

        if path_src_dst_cand != path_src_dst_base:
            raise RuntimeError("Path ordering mismatch between baseline and candidate features (Fermi-Sched).")

        # Convert candidate features to tensors with correct dtypes (int vs float)
        x_tf = fermi_sched_to_tensor_dict(x_cand)

        preds_raw = model(x_tf, training=False).numpy().reshape(-1)
        if preds_raw.shape[0] != n_paths:
            raise RuntimeError("Predictions length mismatch with n_paths (Fermi-Sched).")

        # For jitter, original Fermi predict.py does np.exp(predictions)
        if metric == "jitter":
            preds = np.exp(preds_raw)
        else:
            preds = preds_raw

        header = (
            "  idx src dst            route_nodes    traffic    packets   "
            f"true_{metric}_orig   pred_{metric}"
        )
        print(header)
        print("-" * len(header))
        
        traffic_arr = np.array(x_base["traffic"]).reshape(-1)
        packets_arr = np.array(x_base["packets"]).reshape(-1)

        for i in range(n_paths):
            src_i, dst_i = path_src_dst_cand[i]
            route_str = path_routes_cand[i]

            traffic_i = float(traffic_arr[i])
            packets_i = float(packets_arr[i])
            true_val = float(y_base[i])
            pred_val = float(preds[i])

            mark = "*" if (src_i == c_src and dst_i == c_dst) else " "

            print(
                f"{i:5d} {src_i:3d} {dst_i:3d} {mark}{route_str:>25s} "
                f"{traffic_i:10.3f} {packets_i:10.3f} "
                f"{true_val:13.6f} {pred_val:13.6f}"
            )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mode B predictor for Scheduling (RouteNet-Erlang & RouteNet-Fermi) with candidate routes."
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["erlang", "fermi"],
        help="Which GNN: 'erlang' (RouteNet-Erlang/Scheduling) or 'fermi' (RouteNet-Fermi/scheduling).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="delay",
        choices=["delay", "jitter", "loss"],
        help="Target metric.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Use samples from the training or test scheduling set.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index among *valid* scheduling samples for this framework/split.",
    )
    parser.add_argument(
        "--routes_file",
        type=str,
        required=True,
        help="Text file with candidate routes (e.g., candidate_routes_0_5.txt).",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help=(
            "Optional substring of the topology graph file name to filter samples by, "
            "e.g. 'geant2', 'nsfnet', 'rediris-wfq'. "
            "If set, only samples whose graph file contains this substring are considered, "
            "and sample_index is taken within that filtered subset."
        ),
    )

    args = parser.parse_args()

    if args.framework == "erlang":
        run_erlang_scheduling(args)
    else:
        run_fermi_scheduling(args)


if __name__ == "__main__":
    main()
