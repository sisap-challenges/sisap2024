using JLD2, HDF5, LinearAlgebra, SimilaritySearch
using Downloads: download
using Base.Iterators, Dates

struct NormCostDist <: SemiMetric end

function SimilaritySearch.evaluate(::NormCostDist, u::T, v) where T
    d = zero(eltype(T))
    @inbounds @simd for i in eachindex(u)
        d += u[i] * v[i]
    end

    one(eltype(T)) - d
end

function eval_queries!(dist::SemiMetric, KNN::Vector{KnnResult}, Q::AbstractDatabase, X::AbstractDatabase, r)
    Threads.@threads for qID in eachindex(Q)
        q = Q[qID]
        for (i, objID) in enumerate(r)
            d = SimilaritySearch.evaluate(dist, q, X[i])
            push_item!(KNN[qID], IdWeight(objID, d))
        end
    end
end

function normalize_vectors!(FloatType, X)
    X_ = eltype(X) === FloatType ? X : FloatType.(X)
    #=
    for c in eachcol(X_)
        normalize!(c)
    end
    =#

    MatrixDatabase(X_)
end

"""
    gold_standard(FT, dist; dbname, qname, s, k, outname)

Computes the gold standard of `k` nearest neighbors of dbname and qname (searching in batches of size `s`)
"""
function gold_standard(::Type{FT}, dist; dbname, qname, s, k, outname) where FT
    Q = jldopen(qname) do f
        normalize_vectors!(FT, Matrix{Float32}(f["emb"]))
    end

    knns, dists = compute_knns(FT, dbname, Q, dist, k, s)

    @info "saving $outname"
    jldsave(outname; knns, dists)
end

function compute_knns(::Type{FT}, dbname::String, Q, dist::SemiMetric, k::Integer, s::Integer) where {FT<:AbstractFloat}
    KNN = [KnnResult(k) for _ in eachindex(Q)]

    h5open(dbname) do f
        E = f["emb"]
        @show typeof(E)
        n = size(E, 2)
        
        @info "working on $dbname (size $(n)) -- $(Dates.now())"
        
        for r in Iterators.partition(1:n, s)
            @info "advance $(r) --- step: $(s) -- $(Dates.now())"
            X = normalize_vectors!(FT, E[:, r])
            @info "starting evaluation of $r"
            eval_queries!(dist, KNN, Q, X, r)
        end
    end

    @info "done $(Dates.now()), now saving"

    knns = zeros(Int32, k, length(KNN))
    dists = zeros(Float32, k, length(KNN))

    for (i, res) in enumerate(KNN)
        knns[:, i] .= IdView(res)
        dists[:, i] .= DistView(res)
    end

    knns, dists
end

function main(; qname="data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5", k=1000, s=10^5)
    #dist = NormalizedCosineDistance()
    dist = NormCostDist()
    FloatType = Float32

    for sizename in ["300K", "1M", "10M", "100M"]
        dbname = "data2024/laion2B-en-clip768v2-n=$sizename.h5"
        outname = "data2024/gold-standard-dbsize=$sizename--" * basename(qname)
        if isfile(outname)
            @info "found $outname, skipping to the next setup"
        else
            @info "start $(Dates.now()) $outname"
            gold_standard(FloatType, dist; dbname, qname, s, k, outname)
        end
    end
end
