# random allocation rule
function Rand(nsbj::Int64, seed::Int64 = 314159)
    Random.seed!(seed)

    trt = zeros(Int64, nsbj)
    prb = zeros(Float64, nsbj)

    N1 = 0
    for j in eachindex(trt)
        # calculating probability of treatment assignment
        prb[j] = (0.5*nsbj-N1)/(nsbj-j+1)

        # treatment assignment
        trt[j] = rand(Binomial(1, prb[j]))

        N1 += trt[j]
    end
    
    return trt
end


# truncated binomial design
function tbd(nsbj::Int64, seed::Int64 = 314159)
    Random.seed!(seed)

    trt = zeros(Int64, nsbj)
    prb = zeros(Float64, nsbj)

    N1 = 0
    N2 = 0
    for j in eachindex(trt)
        # calculating probability of treatment assignment
        prb[j] = max(N1, N2) < nsbj/2 ? 0.5 : (N1 < N2 ? 1 : 0)

        # treatment assignment
        trt[j] = rand(Binomial(1, prb[j]))

        N1 += trt[j]
        N2 = j - N1
    end

    return trt
end