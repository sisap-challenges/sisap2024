include("io.jl")
include("laion-gold-standards.jl")

function build_hsp(dbname::AbstractString, qname::AbstractString, gold::AbstractString)
    Q = loadf32(qname) |> MatrixDatabase
    X = loadf32(dbname) |> MatrixDatabase
    knns, dists = load(gold, "knns", "dists")
    dist = NormalizedCosineDistance()

    hsp_queries(dist, X, Q, knns, dists)
end


function loadf32(filename; normalize=true)
    @info "loading $filename"
    Q = load(filename, "emb")
    @assert eltype(Q) == Float32
    if normalize
        for c in eachcol(Q)
            normalize!(c)
        end
    end
    Q
end

function getresname(size, qname, dbname, k)
    outname = "results/$(size)/$(basename(qname))/searchgraph--k=$k--" * basename(dbname)
    outname = replace(outname, ".h5" => "")
    outname * ".h5"
end

function solve_queries_with_cost(size::AbstractString, qname::AbstractString; k=30, outname=nothing, logbase=1.5, minrecall=0.9, disjointbase=1.01e
    dbname = "laion2B-en-clip768v2-n=$size.h5"
    outname = outname === nothing ? getresname(size, qname, dbname, k) : outname
    mkpath(dirname(outname))
     
    Q = loadf32(qname) |> MatrixDatabase
    db = loadf32(dbname) |> MatrixDatabase
    dist = NormalizedCosineDistance()

    G = SearchGraph(; dist, db)
    ctx = SearchGraphContext(
                             neighborhood = Neighborhood(SatNeighborhood(); logbase),
                             hyperparameters_callback = OptimizeParameters(MinRecall(minrecall)),
                             hints_callback = DisjointHints(disjointbase)
                            )

    buildtime = @elapsed index!(G, ctx)
    memory = Base.summarysize(G)
    n = length(Q)
    knns = zeros(Int32, k, n)
    dists = zeros(Float32, k, n)
    searchtime = zeros(Float64, n)
    cost = zeros(Int32, n)
    Threads.@threads :static for i in 1:n
        res = getknnresult(k, ctx)
        t = @elapsed p = search(G, Q[i], res)
        searchtime[i] = t
        cost[i] = p.cost
        k_ = length(res)
        knns[1:k_, i] .= IdView(res)
        dists[1:k_, i] .= DistView(res)
    end

    jldsave(outname; knns, dists, cost, buildtime, searchtime, memory, name="SearchGraph", params="b=$logbase r=$minrecall")
    outname
end


function solve_queries(size::AbstractString, qname::AbstractString; k=30, outname=nothing, logbase=1.5, minrecall=0.95, disjointbase=1.01)
    dbname = "laion2B-en-clip768v2-n=$size.h5"
    outname = outname === nothing ? getresname(size, qname, dbname, k) : outname
    mkpath(dirname(outname))
     
    Q = loadf32(qname) |> MatrixDatabase
    db = loadf32(dbname) |> MatrixDatabase
    dist = NormalizedCosineDistance()

    G = SearchGraph(; dist, db)
    ctx = SearchGraphContext(
                             neighborhood = Neighborhood(SatNeighborhood(); logbase),
                             hyperparameters_callback = OptimizeParameters(MinRecall(minrecall)),
                             hints_callback = DisjointHints(disjointbase)
                            )

    buildtime = @elapsed index!(G, ctx)
    searchtime = @elapsed knns, dists = searchbatch(G, Q, k)
    memory = Base.summarysize(G)

    jldsave(outname; knns, dists, buildtime, searchtime, memory, name="SearchGraph", params="b=$logbase r=$minrecall disjoint=$disjointbase")
    outname
end
