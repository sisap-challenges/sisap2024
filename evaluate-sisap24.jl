using HDF5, JLD2, FileIO, Glob, SimilaritySearch, DataFrames, CSV, OhMyREPL

function evaluate_task(files, gold, D)
    for filename in files
        @info filename
        knns, algo, params, buildtime, querytime = if startswith(filename, "sisap24-example-julia")
            jldopen(filename) do f
                f["knns"], f["algo"], f["params"], f["buildtime"], f["querytime"]
            end
        else
            h5open(filename) do f
                f["knns"][], read_attribute(f, "algo"), read_attribute(f, "params"), read_attribute(f, "buildtime"), read_attribute(f, "querytime")
            end
        end

        recall = macrorecall(view(gold, 1:30, :), knns)
        push!(D, (; algo, params, buildtime, querytime, recall, filename))
    end

    D
end

function eval_task1()
    gold = load("/home/sisap23evaluation/data2024/gold-standard-dbsize=100M--private-queries-2024-laion2B-en-clip768v2-n=10k-epsilon=0.2.h5", "knns")
    D = DataFrame(algo=[], params=[], buildtime=[], querytime=[], recall=[], filename=[])
    deglib = glob("sisap24-deglib/result/100M/uint512/*.h5")
    hiob = glob("sisap24-hiob/result/clip768v2.task1/100M/*/*.h5")
    hsp = glob("sisap24-HSP/result/clip768v2/100M/*.h5")
    lmi = glob("sisap24-lmi/result/task1/100M/*.h5")
    BL = glob("sisap24-example-julia/results-task1/100M/20240825-220654/*.h5") 
    evaluate_task(vcat(deglib, hiob, hsp, lmi, BL), gold, D)
    CSV.write("sisap24-task1.csv", D)
end

function eval_task2()
    gold = load("/home/sisap23evaluation/data2024/gold-standard-dbsize=100M--private-queries-2024-laion2B-en-clip768v2-n=10k-epsilon=0.2.h5", "knns")
    D = DataFrame(algo=[], params=[], buildtime=[], querytime=[], recall=[], filename=[])
    hiob = glob("sisap24-hiob/result/clip768v2/100M/*/*.h5")
    # lmi fails due to wallclock
    BL = glob("sisap24-example-julia/results-task2/100M/20240825-065055/*.h5")
    evaluate_task(vcat(hiob, BL), gold, D)
    CSV.write("sisap24-task2.csv", D)
end

function eval_task3()
    gold = load("/home/sisap23evaluation/data2024/gold-standard-dbsize=100M--private-queries-2024-laion2B-en-clip768v2-n=10k-epsilon=0.2.h5", "knns")
    D = DataFrame(algo=[], params=[], buildtime=[], querytime=[], recall=[], filename=[])
    deglib = glob("sisap24-deglib/result/100M/uint64/*.h5")
    hiob = glob("sisap24-hiob/result/clip768v2.task3/100M/*/*.h5")
    lmi = glob("sisap24-lmi/result/task3/100M/*.h5")
    BL = glob("sisap24-example-julia/results-task3/100M/20240825-154256/*.h5")
    evaluate_task(vcat(deglib, hiob, BL), gold, D)
    CSV.write("sisap24-task3.csv", D)
end

eval_task1()
eval_task2()
eval_task3()

