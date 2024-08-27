using FileIO, HDF5, JLD2

rawgold = "data2024/gold-standard-dbsize=100M--raw-private-queries-2024-laion2B-en-clip768v2-n=12500.h5"
rawqueries = "data2024/private-queries-2024-raw-n=12500-epsilon=0.2.h5"

goldfile = "data2024/gold-standard-dbsize=100M--private-queries-2024-laion2B-en-clip768v2-n=10k-epsilon=0.2.h5"
queriesfile = "data2024/private-queries-2024-laion2B-en-clip768v2-n=10k-epsilon=0.2.h5"

knns, dists = load(rawgold, "knns", "dists")
e = 1e-4
m = 30
n = 10_000
L = findall((dists[m, :] .- dists[m-1, :]) .>= e)
@assert length(L) > n 
resize!(L, n)
@assert sum(dists[29, L] .== dists[30, L]) == 0

jldsave(goldfile; knns=knns[:, L], dists=dists[:, L])
X = load(rawqueries, "emb")
jldsave(queriesfile; emb=X[:, L])
