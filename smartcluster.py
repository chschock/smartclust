import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from collections import defaultdict
from pulp import LpProblem, LpVariable, lpSum, LpInteger, LpMinimize

colors = '#FFC312 #C4E538 #12CBC4 #ED4C67 #F79F1F #A3CB38 #1289A7 #B53471 #EE5A24 ' \
         '#009432 #0652DD #833471 #EA2027 #006266 #1B1464 #6F1E51'.split()


def get_lp(Z, stiffness):
    """
    Define linear program min <c,x> under Ax = b from linkage matrix `Z` such that the
    solution indicates natural cluster link.
    `stiffness` determines how much to enforce clustersize to be half the point set size.
    """
    n_clust = Z.shape[0]
    n_pts = n_clust + 1

    A = np.zeros((n_pts, n_clust + n_pts), dtype=np.int)
    b = np.ones(n_pts, dtype=np.int)
    obj = np.zeros(n_clust + n_pts)
    superdist = dict()
    for l, r, dist, _ in Z:
        superdist[l] = dist
        superdist[r] = dist
    superdist[n_pts + n_clust - 1] = superdist[n_pts + n_clust - 2]  # 0

    for i in range(n_pts):
        A[i, i] = 1
        obj[i] = - superdist[i] * 0

    for i, (l, r, dist, cnt) in enumerate(Z):
        c = i + n_pts
        l, r = int(l), int(r)
        A[:, c] += A[:, l] + A[:, r]
        obj[c] = - (superdist[c] - dist) * cnt ** stiffness * (n_pts - cnt) ** stiffness

    return A, b, obj


def pulp_solve(Z, A, b, obj):
    prob = LpProblem("SmartClust", LpMinimize)
    n_clust = 2 * len(Z) + 1
    clust = LpVariable.dicts("clust", (list(range(n_clust))), 0, 1, LpInteger)
    for i, a in enumerate(A):
        prob += lpSum(clust[i] for i in np.where(a > 0)[0]) == 1
    prob += sum(clust[i] * obj[i] for i in range(n_clust))
    prob.writeLP("SmartClust.lp")
    prob.solve()
    sol = np.array([clust[i].varValue for i in range(n_clust)])
    return sol


def plot_tree(Z, cluster_heads, **dendrogram_params):
    n_pts = len(Z) + 1

    def subtree(c, col):
        c = int(c)
        if c < n_pts:
            return
        link_cols[c] = col
        subtree(Z[c - n_pts, 0], col)
        subtree(Z[c - n_pts, 1], col)

    link_cols = defaultdict(lambda: '#5758BB')
    for i, clh in enumerate(cluster_heads):
        subtree(clh, colors[i % len(colors)])

    return dendrogram(Z=Z, link_color_func=lambda x: link_cols[x])
