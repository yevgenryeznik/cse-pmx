# population-based ANCOVA, given a set of observations simulated for one trial
function p_value_ancova_pop(obs::Population)
    # calculating CHFBL for every subject
    chfbl = @pipe DataFrame(obs) |>
        filter(:SARA => (value -> !ismissing(value)), _) |> 
        groupby(_, :id) |> 
        combine(_, :SARA => (y -> y[end] - y[1]) => :CHFBL)

    # setting up a dataset for analysis    
    obs_ = @pipe DataFrame(obs) |>
        filter(:SARA => (value -> !ismissing(value)), _) |> 
        select(_, :id, :TRT, :TSO, :SARA) |> 
        groupby(_, :id) |> 
        combine(_, first) |>
        innerjoin(_, chfbl, on = :id)

    # performing test
    test = lm(@formula(CHFBL ~ TRT + TSO + SARA), obs_)
    
    # returning test's p-value
    p_value = GLM.coeftable(test).cols[4][2]

    return p_value   
end


# population-based ANCOVA, given observations simulated for many trials
function ancova_pop(obs::Vector{<:Population}; α::Number = 0.05)
    p_value = [p_value_ancova_pop(obs[s]) for s in eachindex(obs)]
    reject = [Int(p <= α) for p in p_value]
    error_rate = mean(reject)

    error_rate
end


# randomization-based ANCOVA, given a set of observations simulated for one trial
function p_value_ancova_rnd(obs::Population, ref::Matrix{Int64}; nseq = 15000, seed = 314159)
    # calculating CHFBL for every subject
    chfbl = @pipe DataFrame(obs) |>
        filter(:SARA => (value -> !ismissing(value)), _) |> 
        groupby(_, :id) |> 
        combine(_, :SARA => (y -> y[end] - y[1]) => :CHFBL)

    # setting up a dataset for analysis    
    obs_ = @pipe DataFrame(obs) |>
        filter(:SARA => (value -> !ismissing(value)), _) |> 
        select(_, :id, :TRT, :TSO, :SARA) |> 
        groupby(_, :id) |> 
        combine(_, first) |>
        innerjoin(_, chfbl, on = :id)

    # selecting re-randomization sequences at random from the reference set 
    Random.seed!(seed)
    seqids = axes(ref, 2)
    trt = ref[:, sample(seqids, nseq)]

    # performing test
    test = lm(@formula(CHFBL ~ TSO + SARA), obs_)
        
    # residuals: observed - prediction 
    R = residuals(test)

    # ranks
    nsbj = nrow(obs_)
    a = 1:nsbj
    ā = mean(a)
    
    # statistics based on ranks
    T = obs_[!, :TRT]
    Sobs = T[sortperm(R)]' * (a .- ā)
    
    # statistics based on rerandomization
    S = [T' * (a .- ā) for T in eachcol(trt[sortperm(R), :])]

    # p-value
    reject = [Int(abs(S[j]) >= abs(Sobs)) for j in eachindex(S)]
    p_value = mean(reject)

    return p_value
end


# randomization-based ANCOVA, given observations simulated for many trials
function ancova_rnd(obs::Vector{<:Population}, ref::Matrix{Int64}; α::Number = 0.05, nseq::Int64 = 15000, seed = 314159)
    p_value = [p_value_ancova_rnd(obs[s], ref, nseq = nseq, seed = seed + 2*s) for s in eachindex(obs)]
    reject = [Int(p .<= α) for p in p_value]
    error_rate = mean(reject)

    return error_rate
end