import numpy as np
from scipy.cluster.hierarchy import dendrogram, to_tree
from collections import defaultdict

colors = '#FFC312 #C4E538 #12CBC4 #ED4C67 #F79F1F #A3CB38 #1289A7 #B53471 #EE5A24 ' \
         '#009432 #0652DD #833471 #EA2027 #006266 #1B1464 #6F1E51'.split()


def flatten(Z, stiffness):
    """
    Flatten hierarchical cluster tree like `get_lp` but explicitly without invoking
    LP solver.

    :param Z: linkage matrix
    :param stiffness: force to push number of custers towards half the link count
    :return: array mapping point id to (hierarchical) cluster id (not consecutive)
    """

    root = to_tree(Z)
    n_links = len(Z)
    id2cluster = [root.id] * (n_links + 1)

    def _score_tree(node, parent_dist):
        """Score nodes and max-aggregate scores on subtrees."""
        nonlocal n_links, id2cluster

        node.score = (parent_dist - node.dist) * \
                     (n_links - node.count) ** stiffness * node.count ** stiffness

        if node.left is None or node.right is None:
            node.max_score = node.score
            id2cluster[node.id] = node.id
            return [node.id]

        sub_ids = [*_score_tree(node.left, node.dist),
                   *_score_tree(node.right, node.dist)]

        sub_sum = node.left.max_score + node.right.max_score
        node.max_score = max(node.score, sub_sum)
        if node.score > sub_sum:
            # overwrite id2cluster with new best cluster node id
            for _id in sub_ids:
                id2cluster[_id] = node.id

        return sub_ids

    _score_tree(root, Z[:, 2].max())

    def _sum_max_scores(node):
        """Traverse tree till flat cluster roots to sum their max_scores."""
        nonlocal id2cluster
        if node.id in id2cluster:
            return node.max_score
        else:
            return _sum_max_scores(node.left) + _sum_max_scores(node.right)

    total_score = _sum_max_scores(root)

    return id2cluster, total_score


def get_lp(Z, stiffness):
    """
    Define linear program min <c,x> under Ax = b from linkage matrix `Z` such that the
    solution indicates natural cluster links.
    `stiffness` determines how much to enforce clustersize to be half the point set size. Values
    greater than 1 make sense.
    Returns definition of linear integer program.
    Remark: The objective function is plausible only up to a certain point. The first term
    describes the length of a branch above a link which is the main heuristic to determine good
    clusters. The second and third term make sure the clustersize is neither too small nor too big.
    Having the strongest boost for clusters of size half the pointset's size may sound arbitrary,
    but turned out to have limited influence - still small clusters are common. You can move the
    mass of this distribution by using 2 different values for stiffness in terms 2 and 3.
    """
    n_clust = 2 * len(Z) + 1
    n_pts = len(Z) + 1

    A = np.zeros((n_pts, n_clust), dtype=np.int)
    b = np.ones(n_pts, dtype=np.int)
    obj = np.zeros(n_clust)
    superdist = dict()
    for l, r, dist, _ in Z:
        superdist[l] = dist
        superdist[r] = dist
    superdist[n_clust - 1] = Z[:, 2].max()  # we will form at least two clusters.

    for i in range(n_pts):
        A[i, i] = 1
        obj[i] = - superdist[i] * (n_pts - 1) ** stiffness

    for i, (l, r, dist, cnt) in enumerate(Z):
        c = i + n_pts
        l, r = int(l), int(r)
        A[:, c] += A[:, l] + A[:, r]
        obj[c] = - (superdist[c] - dist) * cnt ** stiffness * (n_pts - cnt) ** stiffness

    return A, b, obj


def plot_tree(Z, cluster_heads, **dendrogram_params):
    """
    Plot return dendrogram with subtrees colored according to the flat clusters with root ids
    in `cluster_heads`.
    Returns dendrogram.
    """
    n_pts = len(Z) + 1

    def subtree(c, col):  # mark subtree with certain color
        c = int(c)
        if c < n_pts:
            return
        link_cols[c] = col
        subtree(Z[c - n_pts, 0], col)
        subtree(Z[c - n_pts, 1], col)

    link_cols = defaultdict(lambda: '#5758BB')
    for i, clh in enumerate(cluster_heads):
        subtree(clh, colors[i % len(colors)])

    return dendrogram(Z=Z, link_color_func=lambda x: link_cols[x], **dendrogram_params)
