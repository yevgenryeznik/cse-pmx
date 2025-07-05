# population-based t-test, given a single treatment sequence and corresponding responses 
function p_value_t_test_pop(T::Vector{Int}, Y::Vector{Float64})
    # responses in TRT groups
    Y1 = Y[T .== 1]
    Y0 = Y[T .== 0]

    # performing test
    test = EqualVarianceTTest(Y1, Y0)
  
    # returning test's p-value
    return HypothesisTests.pvalue(test)
end


# population-based t-test, given a set of observations simulated for one trial
function p_value_t_test_pop(obs::Population)
    # calculating CHFBL for every subject
    obs_ = @pipe DataFrame(obs) |>
        filter(:SARA => (value -> !ismissing(value)), _) |> 
        groupby(_, [:id, :TRT]) |> 
        combine(_, :SARA => (y -> y[end] - y[1]) => :CHFBL)

    # treatments assignments
    T = Int.(obs_[!, :TRT])

    # responses
    Y = float(obs_[!, :CHFBL])

    # calculating p-value
    p_value = p_value_t_test_pop(T, Y)

    return p_value
end


# population-based t-test, given observations simulated for many trials
function t_test_pop(obs::Vector{<:Population}, α::Number = 0.05)
    p_value = [p_value_t_test_pop(obs[s]) for s in eachindex(obs)]
    reject = [Int(p .<= α) for p in p_value]
    error_rate = mean(reject)

    return error_rate
end


# randomization t-test, given a single treatment sequence and corresponding responses 
function p_value_t_test_rnd(obs::Population, ref::Matrix{Int64}; nseq = 15000, seed = 314159)
    # calculating CHFBL for every subject
    obs_ = @pipe DataFrame(obs) |>
        filter(:SARA => (value -> !ismissing(value)), _) |> 
        groupby(_, [:id, :TRT]) |> 
        combine(_, :SARA => (y -> y[end] - y[1]) => :CHFBL)

    # treatments assignments
    T = Int.(obs_[!, :TRT])

    # responses
    Y = float(obs_[!, :CHFBL])

    # selecting re-randomization sequences at random from the reference set
    Random.seed!(seed) 
    seqids = axes(ref, 2)
    trt = ref[:, sample(seqids, nseq)]
    
    # residuals: observed - prediction 
    R = Y .- mean(Y)

    # ranks
    nsbj = length(T)
    a = 1:nsbj
    ā = mean(a)
    
    # statistics based on ranks
    Sobs = T[sortperm(R)]' * (a .- ā)
    
    # statistics based on rerandomization
    S = [T' * (a .- ā) for T in eachcol(trt[sortperm(R), :])]

    # p-value
    reject = [Int(abs(S[j]) >= abs(Sobs)) for j in eachindex(S)]
    p_value = mean(reject)

    return p_value
end


# randomization t-test, given observations simulated for many trials
function t_test_rnd(obs::Vector{<:Population}, ref::Matrix{Int64}; α::Number = 0.05, nseq::Int64 = 15000, seed = 314159)
    p_value = [p_value_t_test_rnd(obs[s], ref, nseq = nseq, seed = seed + 2*s) for s in eachindex(obs)]
    reject = [Int(p .<= α) for p in p_value]
    error_rate = mean(reject)

    return error_rate
end

