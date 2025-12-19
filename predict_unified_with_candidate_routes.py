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


def resolve_dataset_dir(framework, traffic_mode, split, topo):
    """
    Map (framework, traffic_mode, split, topo) -> directory with tar.gz + graphs + routings.
    Matches the actual folder names under data/traffic_models.
    """
    if framework == "erlang":
        root = os.path.join(ERLANG_DIR, "data", "traffic_models", traffic_mode)
    else:
        root = os.path.join(FERMI_DIR, "data", "traffic_models", traffic_mode)

    split = split.lower()
    topo = topo.lower()
    traffic_mode = traffic_mode.lower()

    # Map traffic_mode to suffix used in folder names
    suffix_map = {
        "all_multiplexed": "multiplexed",
        "autocorrelated": "autocorrelated",
        "constant_bitrate": "constant",
        "modulated": "modulated",
        "onoff": "onoff",
    }

    if traffic_mode not in suffix_map:
        raise SystemExit(f"Unknown traffic_mode '{traffic_mode}' in resolve_dataset_dir().")

    suffix = suffix_map[traffic_mode]

    if split == "test":
        # Test sets are always GBN
        if topo != "gbn":
            raise SystemExit("For split='test', topology must be 'gbn'.")
        # e.g. gbn-multiplexed, gbn-constant, gbn-autocorrelated, ...
        return os.path.join(root, "test", f"gbn-{suffix}")
    else:
        # Train: geant2 / nsfnet
        if topo not in ("geant2", "nsfnet"):
            raise SystemExit("For split='train', topology must be 'geant2' or 'nsfnet'.")
        # e.g. geant2-multiplexed, geant2-constant, nsfnet-autocorrelated, ...
        return os.path.join(root, "train", f"{topo}-{suffix}")



def validate_candidate_path_for_sample(sample, cand_id, path_nodes, framework_tag, args):
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
                f"  dataset_split = {args.dataset_split}\n"
                f"  topology      = {args.topology}\n"
                f"  traffic_mode  = {args.traffic_mode}\n"
                f"  routes_file   = {args.routes_file}\n"
                f"Make sure the routes_file was generated for THIS topology."
            )

# =============================================================================
# ERLANG BRANCH
# =============================================================================

def erlang_import_modules():
    """
    Add RouteNet-Erlang/TrafficModels to sys.path and import Erlang modules.
    """
    tm_dir = os.path.join(ERLANG_DIR, "TrafficModels")
    if tm_dir not in sys.path:
        sys.path.insert(0, tm_dir)

    from datanetAPI import DatanetAPI
    from read_dataset import network_to_hypergraph
    from model import GNN_Model

    return DatanetAPI, network_to_hypergraph, GNN_Model


def erlang_build_graph_and_features(sample, label, network_to_hypergraph):
    """
    Same logic as your working Erlang scripts: build hypergraph representation and features.
    """
    HG = network_to_hypergraph(sample=sample)

    n_p = 0
    n_l = 0
    mapping = {}
    for entity in list(HG.nodes()):
        if entity.startswith("p"):
            mapping[entity] = f"p_{n_p}"
            n_p += 1
        elif entity.startswith("l"):
            mapping[entity] = f"l_{n_l}"
            n_l += 1

    D_G = nx.relabel_nodes(HG, mapping)

    # Incidence arrays
    link_to_path = []
    path_ids = []
    sequence_path = []
    for i in range(n_p):
        node_name = f"p_{i}"
        seq_len = 0
        for elem in D_G[node_name]:
            link_to_path.append(int(elem.replace("l_", "")))
            seq_len += 1
        path_ids.extend(np.full(seq_len, i, dtype=np.int32))
        sequence_path.extend(range(seq_len))

    path_to_link = []
    sequence_links = []
    for i in range(n_l):
        node_name = f"l_{i}"
        seq_len = 0
        for elem in D_G[node_name]:
            path_to_link.append(int(elem.replace("p_", "")))
            seq_len += 1
        sequence_links.extend(np.full(seq_len, i, dtype=np.int32))

    # Skip logic: if any path has 0 jitter or 0 delay, sample is skipped in training
    jitter_vals = list(nx.get_node_attributes(D_G, "jitter").values())
    delay_vals = list(nx.get_node_attributes(D_G, "delay").values())
    if 0 in jitter_vals or 0 in delay_vals:
        raise ValueError("Sample has zero jitter or delay and is skipped in original Erlang dataset.")

    # Path-level features
    traffic = [D_G.nodes[f"p_{i}"]["traffic"] for i in range(n_p)]
    packets = [D_G.nodes[f"p_{i}"]["packets"] for i in range(n_p)]
    time_dist_params = [D_G.nodes[f"p_{i}"]["time_dist_params"] for i in range(n_p)]
    label_vals = [D_G.nodes[f"p_{i}"][label] for i in range(n_p)]

    # Link-level features
    capacity = [D_G.nodes[f"l_{i}"]["capacity"] for i in range(n_l)]

    x = {
        "traffic": np.array(traffic, dtype=np.float32),
        "packets": np.array(packets, dtype=np.float32),
        "time_dist_params": np.array(time_dist_params, dtype=np.float32),
        "capacity": np.array(capacity, dtype=np.float32),
        "link_to_path": np.array(link_to_path, dtype=np.int32),
        "path_to_link": np.array(path_to_link, dtype=np.int32),
        "path_ids": np.array(path_ids, dtype=np.int32),
        "sequence_links": np.array(sequence_links, dtype=np.int32),
        "sequence_path": np.array(sequence_path, dtype=np.int32),
        "n_links": np.array(n_l, dtype=np.int32),
        "n_paths": np.array(n_p, dtype=np.int32),
    }

    y = np.array(label_vals, dtype=np.float32)

    # Per-path src/dst + route string using routing matrix
    R = sample.get_routing_matrix()
    path_src_dst = []
    path_routes = []
    for i in range(n_p):
        src = D_G.nodes[f"p_{i}"]["source"]
        dst = D_G.nodes[f"p_{i}"]["destination"]
        path_src_dst.append((src, dst))
        path = R[src, dst]
        if path is None or len(path) == 0:
            route_str = "(no path)"
        else:
            route_str = "->".join(str(node) for node in path)
        path_routes.append(route_str)

    return x, y, path_src_dst, path_routes, D_G


def erlang_get_sample_features(dataset_dir, label, sample_index, DatanetAPI, network_to_hypergraph):
    """
    Iterate over DatanetAPI and return the sample_index-th valid (after skip logic) sample.
    """
    tool = DatanetAPI(dataset_dir, shuffle=False)
    it = iter(tool)
    valid_idx = -1

    while True:
        try:
            sample = next(it)
        except StopIteration:
            raise RuntimeError(
                f"[Erlang] Reached end of dataset before index {sample_index}. "
                f"Found {valid_idx + 1} valid samples."
            )

        try:
            x, y, path_src_dst, path_routes, D_G = erlang_build_graph_and_features(
                sample, label, network_to_hypergraph
            )
        except ValueError:
            continue  # skip

        valid_idx += 1
        if valid_idx == sample_index:
            return sample, x, y, path_src_dst, path_routes


def erlang_build_features_with_override(sample, label, override_dict, network_to_hypergraph):
    """
    Override routing for some (src,dst) pairs, rebuild features, restore routing.
    """
    R_orig = sample.get_routing_matrix().copy()
    R_mod = R_orig.copy()
    for (s, d), path in override_dict.items():
        R_mod[s, d] = path
    sample._set_routing_matrix(R_mod)

    try:
        x, y, path_src_dst, path_routes, D_G = erlang_build_graph_and_features(
            sample, label, network_to_hypergraph
        )
    finally:
        sample._set_routing_matrix(R_orig)

    return x, y, path_src_dst, path_routes


def erlang_transformation(x, y):
    """
    Same normalization & log transform as Erlang delay script.
    """
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


def erlang_denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100.0


def erlang_load_best_model(params, exp_dir, GNN_Model_cls):
    """
    Pick the best checkpoint by filename MRE and load weights.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(params["HYPERPARAMETERS"]["learning_rate"])
    )
    model = GNN_Model_cls(params)
    loss_object = tf.keras.losses.MeanSquaredError()

    model.compile(
        loss=loss_object,
        optimizer=optimizer,
        run_eagerly=False,
        metrics=erlang_denorm_MAPE,
    )

    ckpt_dir = os.path.join(exp_dir, "ckpt_dir")
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
        raise RuntimeError(f"[Erlang] No checkpoint file found in {ckpt_dir}")

    print(f"BEST CHECKPOINT FOUND (Erlang): {best}")
    model.load_weights(os.path.join(ckpt_dir, best))
    return model


def run_erlang(args):
    import configparser

    DatanetAPI, network_to_hypergraph, GNN_Model = erlang_import_modules()

    label = args.metric.lower()
    if label == "loss":
        raise SystemExit("RouteNet-Erlang in this script supports 'delay' and 'jitter', not 'loss'.")

    dataset_dir = resolve_dataset_dir(
        framework="erlang",
        traffic_mode=args.traffic_mode,
        split=args.dataset_split,
        topo=args.topology,
    )

    metric_cap = label.capitalize()  # 'Delay' or 'Jitter'
    exp_dir = os.path.join(ERLANG_DIR, "TrafficModels", metric_cap, args.traffic_mode)

    print(f"\n=== ERLANG MODE B ===")
    print(f"Dataset directory : {dataset_dir}")
    print(f"Experiment dir    : {exp_dir}")
    print(f"Metric / label    : {label}")
    print(f"Sample index      : {args.sample_index}")
    print(f"Routes file       : {args.routes_file}\n")

    params = configparser.ConfigParser()
    params._interpolation = configparser.ExtendedInterpolation()
    params.read(os.path.join(exp_dir, "config.ini"))

    sample, x_base, y_base, path_src_dst_base, path_routes_base = erlang_get_sample_features(
        dataset_dir, label, args.sample_index, DatanetAPI, network_to_hypergraph
    )

    n_paths = int(x_base["n_paths"])
    print(f"Loaded baseline sample: n_paths = {n_paths}, n_links = {int(x_base['n_links'])}")

    model = erlang_load_best_model(params, exp_dir, GNN_Model)

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

    np.set_printoptions(precision=6, suppress=True, linewidth=140)

    for cand in candidates:
        cand_id = cand["id"]
        c_src = cand["src"]
        c_dst = cand["dst"]
        path_nodes = cand["nodes"]

        # Validate candidate path against this topology
        validate_candidate_path_for_sample(
            sample, cand_id, path_nodes, framework_tag="Erlang", args=args
        )

        print("\n" + "=" * 100)
        print(f"[Erlang] Candidate {cand_id}: src={c_src}, dst={c_dst}, path = {'->'.join(map(str, path_nodes))}")
        print("=" * 100)

        override = {(c_src, c_dst): path_nodes}
        x_cand, _, path_src_dst_cand, path_routes_cand = erlang_build_features_with_override(
            sample, label, override, network_to_hypergraph
        )

        if path_src_dst_cand != path_src_dst_base:
            raise RuntimeError("Path ordering mismatch between baseline and candidate features (Erlang).")

        x_tf = {k: tf.convert_to_tensor(v) for k, v in x_cand.items()}
        y_tf = tf.convert_to_tensor(y_base)
        x_tf_norm, _ = erlang_transformation(x_tf, y_tf)

        preds_log = model(x_tf_norm, training=False)
        preds = np.exp(preds_log.numpy()).reshape(-1)

        if preds.shape[0] != n_paths:
            raise RuntimeError("Predictions length mismatch with n_paths (Erlang).")

        header = (
            "  idx src dst            route_nodes    traffic    packets   "
            f"time_params[0:4] true_{label}_orig   pred_{label}"
        )
        print(header)
        print("-" * len(header))

        for i in range(n_paths):
            src_i, dst_i = path_src_dst_cand[i]
            route_str = path_routes_cand[i]

            traffic_i = float(x_base["traffic"][i])
            packets_i = float(x_base["packets"][i])
            tparams = x_base["time_dist_params"][i][:4]
            true_val = float(y_base[i])
            pred_val = float(preds[i])

            mark = "*" if (src_i == c_src and dst_i == c_dst) else " "

            print(
                f"{i:5d} {src_i:3d} {dst_i:3d} {mark}{route_str:>25s} "
                f"{traffic_i:10.3f} {packets_i:10.3f} {str(tparams):>20s} "
                f"{true_val:13.6f} {pred_val:13.6f}"
            )

# =============================================================================
# FERMI BRANCH
# =============================================================================

def fermi_import_modules(metric):
    """
    Add RouteNet-Fermi/traffic_models to sys.path and import metric-specific
    model + data_generator.
    """
    tm_dir = os.path.join(FERMI_DIR, "traffic_models")
    if tm_dir not in sys.path:
        sys.path.insert(0, tm_dir)
    if FERMI_DIR not in sys.path:
        sys.path.insert(0, FERMI_DIR)

    from datanetAPI import DatanetAPI

    metric = metric.lower()
    if metric == "delay":
        from delay_model import RouteNet_Fermi as FermiModel
        from delay.data_generator import network_to_hypergraph, hypergraph_to_input_data
        subdir = "delay"
    elif metric == "jitter":
        from jitter_model import RouteNet_Fermi as FermiModel
        from jitter.data_generator import network_to_hypergraph, hypergraph_to_input_data
        subdir = "jitter"
    elif metric == "loss":
        from loss_model import RouteNet_Fermi as FermiModel
        from losses.data_generator import network_to_hypergraph, hypergraph_to_input_data
        subdir = "losses"
    else:
        raise SystemExit("Unsupported Fermi metric (should not happen).")

    return DatanetAPI, FermiModel, network_to_hypergraph, hypergraph_to_input_data, subdir


def fermi_build_graph_and_features(sample, network_to_hypergraph, hypergraph_to_input_data):
    """
    Build RouteNet-Fermi hypergraph + inputs for a single sample.
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


def fermi_get_sample_features(dataset_dir, sample_index, metric,
                              DatanetAPI, network_to_hypergraph, hypergraph_to_input_data):
    """
    Iterate over DatanetAPI and return sample_index-th *valid* sample.
    Skip logic:
      - delay:   all labels > 0
      - jitter:  all labels >= 0
      - loss:    all labels >= 0
    """
    tool = DatanetAPI(dataset_dir, shuffle=False)
    it = iter(tool)
    valid_idx = -1
    metric = metric.lower()

    while True:
        try:
            sample = next(it)
        except StopIteration:
            raise RuntimeError(
                f"[Fermi] Reached end of dataset before index {sample_index}. "
                f"Found {valid_idx + 1} valid samples."
            )

        x_dict, y, path_src_dst, path_routes = fermi_build_graph_and_features(
            sample, network_to_hypergraph, hypergraph_to_input_data
        )

        if metric == "delay":
            if not np.all(y > 0):
                continue
        else:
            if not np.all(y >= 0):
                continue

        valid_idx += 1
        if valid_idx == sample_index:
            return sample, x_dict, y, path_src_dst, path_routes


def fermi_build_features_with_override(sample, override_dict,
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
        x_dict, y, path_src_dst, path_routes = fermi_build_graph_and_features(
            sample, network_to_hypergraph, hypergraph_to_input_data
        )
    finally:
        sample._set_routing_matrix(R_orig)

    return x_dict, path_src_dst, path_routes


def fermi_to_tensor_dict(x_dict):
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


def fermi_load_best_model(ckpt_dir, FermiModel_cls, example_x):
    """
    Build and load RouteNet-Fermi model from ckpt_dir_* by choosing min-MRE ckpt.
    """
    model = FermiModel_cls()

    # Build variables by a forward pass using correct dtypes
    x_tf = fermi_to_tensor_dict(example_x)
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
        raise RuntimeError(f"[Fermi] No checkpoint file found in {ckpt_dir}")

    print(f"BEST CHECKPOINT FOUND (Fermi): {best}")
    model.load_weights(os.path.join(ckpt_dir, best))
    return model



def run_fermi(args):
    metric = args.metric.lower()
    DatanetAPI, FermiModel, network_to_hypergraph, hypergraph_to_input_data, subdir = (
        fermi_import_modules(metric)
    )

    dataset_dir = resolve_dataset_dir(
        framework="fermi",
        traffic_mode=args.traffic_mode,
        split=args.dataset_split,
        topo=args.topology,
    )

    # Match Fermi's original code:
    #  - jitter: ckpt_dir_mape_<traffic_mode>
    #  - delay/loss: ckpt_dir_<traffic_mode>
    if metric == "jitter":
        ckpt_dir_name = f"ckpt_dir_mape_{args.traffic_mode}"
    else:
        ckpt_dir_name = f"ckpt_dir_{args.traffic_mode}"

    ckpt_dir = os.path.join(
        FERMI_DIR,
        "traffic_models",
        subdir,
        ckpt_dir_name,
    )

    print(f"\n=== FERMI MODE B (metric={metric}) ===")
    print(f"Dataset directory : {dataset_dir}")
    print(f"Checkpoint dir    : {ckpt_dir}")
    print(f"Sample index      : {args.sample_index}")
    print(f"Routes file       : {args.routes_file}\n")

    sample, x_base, y_base, path_src_dst_base, path_routes_base = fermi_get_sample_features(
        dataset_dir,
        args.sample_index,
        metric,
        DatanetAPI,
        network_to_hypergraph,
        hypergraph_to_input_data,
    )

    n_paths = len(y_base)
    print(f"Loaded baseline sample (Fermi): n_paths = {n_paths}")

    model = fermi_load_best_model(ckpt_dir, FermiModel, x_base)

    candidates = parse_candidate_routes(args.routes_file)
    if not candidates:
        raise RuntimeError(f"No candidate routes parsed from {args.routes_file}")

    first_src = candidates[0]["src"]
    first_dst = candidates[0]["dst"]
    for c in candidates:
        if c["src"] != first_src or c["dst"] != first_dst:
            print("WARNING: Not all candidates share the same (src,dst); interpretation is trickier.")
            break

    np.set_printoptions(precision=6, suppress=True, linewidth=140)

    for cand in candidates:
        cand_id = cand["id"]
        c_src = cand["src"]
        c_dst = cand["dst"]
        path_nodes = cand["nodes"]

        print("\n" + "=" * 100)
        print(f"[Fermi] Candidate {cand_id}: src={c_src}, dst={c_dst}, path = {'->'.join(map(str, path_nodes))}")
        print("=" * 100)

        override = {(c_src, c_dst): path_nodes}
        x_cand, path_src_dst_cand, path_routes_cand = fermi_build_features_with_override(
            sample,
            override,
            network_to_hypergraph,
            hypergraph_to_input_data,
        )

        if path_src_dst_cand != path_src_dst_base:
            raise RuntimeError("Path ordering mismatch between baseline and candidate features (Fermi).")

        # Convert candidate features to tensors with correct dtypes (int vs float)
        x_tf = fermi_to_tensor_dict(x_cand)

        preds_raw = model(x_tf, training=False).numpy().reshape(-1)
        if preds_raw.shape[0] != n_paths:
            raise RuntimeError("Predictions length mismatch with n_paths (Fermi).")

        # For jitter, original Fermi predict.py does np.exp(predictions)
        if metric == "jitter":
            preds = np.exp(preds_raw)
        else:
            preds = preds_raw

        label = metric

        header = (
            "  idx src dst            route_nodes    traffic    packets   "
            f"true_{label}_orig   pred_{label}"
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
        description="Unified Mode B predictor for RouteNet-Erlang and RouteNet-Fermi with candidate routes."
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["erlang", "fermi"],
        help="Which GNN: 'erlang' (RouteNet-Erlang) or 'fermi' (RouteNet-Fermi).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="delay",
        choices=["delay", "jitter", "loss"],
        help="Target metric.",
    )
    parser.add_argument(
        "--traffic_mode",
        type=str,
        default="all_multiplexed",
        choices=["all_multiplexed", "autocorrelated", "constant_bitrate", "modulated", "onoff"],
        help="Traffic model under data/traffic_models.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Use samples from the training or test set.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="gbn",
        choices=["gbn", "geant2", "nsfnet"],
        help="gbn for test; geant2/nsfnet for train.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index among *valid* samples for this framework/split/topology.",
    )
    parser.add_argument(
        "--routes_file",
        type=str,
        required=True,
        help="Text file with candidate routes (e.g., candidate_routes_0_5.txt).",
    )

    args = parser.parse_args()

    if args.framework == "erlang":
        run_erlang(args)
    else:
        run_fermi(args)


if __name__ == "__main__":
    main()
