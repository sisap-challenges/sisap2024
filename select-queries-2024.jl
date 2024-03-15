using MultivariateStats, Parquet2, JLD2, HDF5, LinearAlgebra, SimilaritySearch
using Downloads: download
using Base.Iterators, Dates, Random


include("io.jl")
include("laion-gold-standards.jl")

function select_queries(;
        queriesfromfile="img_emb_1000.jld2",
        epsilon=1/3,
        n = 10^5
        #queriesfromfile="/data1/sadit/laion2B-en/img_emb/img_emb_1000.jld2"
    )
    
    Q = Matrix{Float32}(load(queriesfromfile, "emb"))
    @show queriesfromfile size(Q)
    dist = CosineDistance()
    db = VectorDatabase(Vector{Float32}[])
    G = SearchGraph(; dist, db)
    ctx = SearchGraphContext(
                             getcontext(G);
                             hyperparameters_callback=OptimizeParameters(MinRecall(0.9)),
                             neighborhood=Neighborhood(SatNeighborhood(); logbase=1.5)
                            )

    S = neardup(G, ctx, MatrixDatabase(Q), epsilon)
    ilist = unique(S.nn); shuffle!(ilist); ilist = ilist[1:n]
    Q[:, ilist]
end

