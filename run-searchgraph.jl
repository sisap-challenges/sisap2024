using HDF5, FileIO, SimilaritySearch, JLD2, LinearAlgebra, DataFrames, Glob

function load_database(fname, key, normalize::Bool)
    @info "loading $fname"
    @time X = load(fname, key)
    @assert eltype(X) == Float32
    normalize && for c in eachcol(X) normalize!(c) end
    X
end

function run_recall(goldfile, resfiles::Vector; k=30, k2=k)
    g = load(goldfile)
    D = DataFrame(name=String[], params=String[], buildtime=Float64[], searchtime=Float64[], recall=Float64[])

    for resfile in resfiles
        r = load(resfile)
        A = [Set(c) for c in eachcol(g["knns"][1:k, :])]
        B = [Set(c) for c in eachcol(r["knns"][1:k2, :])]
        #recall = macrorecall(g["knns"][1:k, :], r["knns"][1:k2, :])
        recall = macrorecall(A, B)
        push!(D, (r["name"], r["params"], r["buildtime"], r["searchtime"], recall))
    end

    D
end

function run_search(;
        dfile = "SISAP23-Challenge/laion2B-en-clip768v2-n=1M.h5",
        qfile = "private-queries-gold-10k-clip768v2.h5",
        key = "emb",
        k = 30,
        minrecall = 0.9,
        dist = TurboNormalizedCosineDistance(),
        normalize = true,
        hints_callback = KCentersHints(), #EpsilonHints(quantile=0.1), # DisjointHints()
        neighborhood = Neighborhood(SatNeighborhood(); logbase=1.5),
        outdir = "results"
    )
    db = load_database(dfile, key, normalize) |> StrideMatrixDatabase
    queries = load_database(qfile, key, normalize) |> StrideMatrixDatabase
    G = SearchGraph(; dist, db)
    ctx = SearchGraphContext(getcontext(G);
                             hints_callback,
                             hyperparameters_callback=OptimizeParameters(MinRecall(minrecall)),
                             neighborhood
                            )
    buildtime = @elapsed index!(G, ctx)
    buildtime += @elapsed optimize_index!(G, ctx, MinRecall(0.9))
    name = "SearchGraph"

    f = 1.05
    G.search_algo.Δ /= f^3
    outdir = let
        name_ = joinpath(outdir, basename(dfile))
        replace(name_, ".h5" => "")
    end
    mkpath(outdir)
    
    for i in 1:15
        params = "Delta=$(round(G.search_algo.Δ, digits=3)) $(string(hints_callback))"
        searchtime = @elapsed knns, dists = searchbatch(G, ctx, queries, k)
        resname = let
            resname = "$name--$params--k=$k.h5"
            resname = replace(resname, r"\(|\)|\{|\}|\[|\]|\\|/|\s" => "_")
            resname = joinpath(outdir, resname)
        end
        @info "saving $resname"
        jldsave(resname; knns, dists, searchtime, buildtime, params, name)
        G.search_algo.Δ *= f
    end

    G
end
