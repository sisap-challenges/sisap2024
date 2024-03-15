using UnicodePlots, LinearAlgebra, HDF5

N = h5open("data2024/laion2B-en-clip768v2-n=10M.h5") do f
    X = f["emb"]
    m, n = size(X)
    @show m, n
    vcat([norm(c) for c in eachcol(X[:, 1:1000])], [norm(c) for c in eachcol(X[:, end-1000:end])])
end

histogram(N)


