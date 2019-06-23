# SmartClust

## TL;DR

Here you find a method to make good flat clusters from a hierarchical clustering dendrogram,
so you choose the clustering parameters and the method chooses flat clusters that for me turned
out to be better than the standard techniques.

## Introduction

Hierarchical clustering belongs to the most useful clustering algorithms, as a dendrogram can represent your data in a much more informative way than pure flat clusters. The complexity of the algorithms is infeasible for larger datasets n_points>>10000, but you can probably get along by combining local clustering with the hierarchical clustering for the last step. Despite representing your data in a complex manner, hierarchical clustering works with all (?) metrics and many different linkage methods to define the distance between clusters. Some of the latter depend on a euclidean metric though. These linkage method in scipy have names like 'single', 'complete', 'average', 'weighted', 'ward'.

To understand the explanation below you probably have to read up a bit on hierarchical clustering e.g. from https://en.wikipedia.org/wiki/Hierarchical_clustering and have a look at the cluster trees of the jupyter notebook.

When presenting your data you do probably want flat clusters, as tree like visualizations are hard to grasp. The dendrogram of hierarchical clustering is a nice point to start from for forming flat clusters, because it is optimal in a certain sense (by the parameters you chose) and it gives you global information about the datas structure. In the simplest case cutting the tree at a certain hight already does it. If the data is not so easily separable, you might want a more sophisticated method. Among the standard choices in scipy is choosing a statistics for each link that is monotonous up the tree (to the root). For example the inconsistency criterion works fine - but I didn't get it to give satisfactory results for data of higher variance - a set of documents retrieved by a nearest neighbour query.

There is a natural intuition for determining good clusters: the length of the branch above a link. The longer it is, the more the group of points below the link separates from the sibling. What if we use this criterion for a global optimization, and push it into acceptable boundaries for the cluster sizes?

## Solution as linear integer program

The feasible set is the set of all combinations of branches, that make up a clustering, which is those that don't have clusters included in clusters (partial overlaps are excluded by the tree structure). Linear integer programming is a natural choice for the optimization, with the binary solutions marking the chosen cluster links. Solvers for this type of problems are not terribly efficient, so we don't want to go beyond linear optimization.

Thus the problem structure is:
minimize <c, x> subject to A x = 1 and x_i >= 0 for all i, where 1 is the vector of all 1 components.
A is a matrix of size n_points x n_clusters, where n_clusters is always 2 n_points - 1 due to the nature of the dendrogram, a binary tree.
The right hand side of Ax=1 means 'a point is in exactly one cluster'.

Setting up the feasible area with is not very complicated. You design the matrix for Ax=b starting with the variables on the bottom and recursively up to the top.
The coefficients of the objective function that turned out to do what I wanted have this form:
c(x_i) = branch_length_above_link(x_i) * size(x_i) ** alpha * (n_points - size(x_i)) ** alpha,  alpha ~ 1.7
where x is a vector with components indicating potential clusters, i.e. we optimize
\sum_i c(x_i) * x_i.
The second and third factor do the job pushing cluster size towards 1/2 the point sets size, which seems a bit arbitrary but does the job.

## Remarks

### Performance
The scipy linear program solver is too slow for problems of size 100 already (1 sec), so we use an external one through the pulp package.

### Sample data
The notebook does nearest neighbour search on vectorizations of wikipedia articles. Those vectorizations are derived from word embeddings. The data is included as such, because it would take some effort to create random data with realistic properties. The data is not easily separable due to its similarity, as it comes from a similarity search.

### when you don't need this
If you have clearly separable data, which you see by long vertical branches in your dendrogram, like half it's total height, then you probably don't need a smart method to determine flat clusters.
