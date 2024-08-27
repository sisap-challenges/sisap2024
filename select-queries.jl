using Parquet2, JLD2, HDF5, LinearAlgebra, SimilaritySearch
using Downloads: download
using Base.Iterators, Dates, Random


include("io.jl")

function select_queries(;
        epsilon::Float64, ## 1/3
        n::Int,
        dbfile
    )
    
    @info "selecting queries from $dbfile"
    Q = Matrix{Float32}(load(dbfile, "emb"))
    @info "loaded $(size(Q)) matrix, selecting $n vectors with epsilon $epsilon"
    dist = CosineDistance()
    db = VectorDatabase(Vector{Float32}[])
    G = SearchGraph(; dist, db)
    ctx = SearchGraphContext(
                             getcontext(G);
                             hyperparameters_callback=OptimizeParameters(MinRecall(0.9)),
                             neighborhood=Neighborhood(SatNeighborhood(); logbase=1.25)
                            )

    S = neardup(G, ctx, MatrixDatabase(Q), epsilon)
    ilist = unique(S.nn); shuffle!(ilist); ilist = ilist[1:n]
    Q[:, ilist]
end

function main()
    epsilon = 0.2
    n = 12500
    dbfile = "/data1/sadit/laion2B-en/img_emb/img_emb_1001.jld2"
    qfile = "data2024/private-queries-2024-raw-n=$n-epsilon=$epsilon.h5"
    if !isfile(qfile)
        Q = select_queries(; epsilon, n, dbfile)
        jldsave(qfile; emb=Q)
    end
    
end

