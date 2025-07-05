# population-based MMRM, given a set of observations simulated for one trial
function p_value_mmrm_sara_pop(obs::Population)
    # setting up a dataset for analysis    
    obs_ = @pipe DataFrame(obs) |>
        filter(:SARA => x -> !ismissing(x), _) |>
        select(_, :id, :time, :TSO, :SARA, :TRT) |>
        groupby(_, :id) |>
        transform(_, 
            :time => (y -> CategoricalArray([i == 1 ? "BL" : "Visit$(i-1)" for i in eachindex(y)])) => :VISIT,
            :TRT => categorical => :TRTc,
            :id => categorical => :id,
            :SARA => (y -> y .- y[1]) => :CHFBL,
            :SARA => (y -> y[1]) => :SARABL
        )

    # performing MMRM fit
    mmrm_formula = @formula(SARA ~ TSO + (SARABL + TRTc) * VISIT + (1 | id))  
    mmrm = fit(LinearMixedModel, mmrm_formula, obs_)
    
    # returning test's p-value
    p_value = mmrm.pvalues[end]
    
    return p_value   
end


function p_value_mmrm_sara_chfbl_pop(obs::Population)
    # setting up a dataset for analysis    
    obs_ = @pipe DataFrame(obs) |>
        filter(:SARA => x -> !ismissing(x), _) |>
        select(_, :id, :time, :TSO, :SARA, :TRT) |>
        groupby(_, :id) |>
        transform(_, 
            :time => (y -> CategoricalArray([i == 1 ? "BL" : "Visit$(i-1)" for i in eachindex(y)])) => :VISIT,
            :TRT => categorical => :TRTc,
            :id => categorical => :id,
            :SARA => (y -> y .- y[1]) => :CHFBL,
            :SARA => (y -> y[1]) => :SARABL
        ) |>
        filter(:VISIT => (value -> value != "BL"), _) 

    # performing MMRM fit
    mmrm_formula = @formula(CHFBL ~ TSO + (SARABL + TRTc) * VISIT + (1 | id))    
    mmrm = fit(LinearMixedModel, mmrm_formula, obs_)

    # returning test's p-value
    p_value = mmrm.pvalues[end]
    
    return p_value   
end


# population-based MMRM, given observations simulated for many trials
function mmrm_sara_pop(obs::Vector{<:Population}; α::Number = 0.05)
    p_value = [p_value_mmrm_sara_pop(obs[s]) for s in eachindex(obs)]
    reject = [Int(p <= α) for p in p_value]
    error_rate = mean(reject)

    error_rate
end

function mmrm_sara_chfbl_pop(obs::Vector{<:Population}; α::Number = 0.05)
    p_value = [p_value_mmrm_sara_chfbl_pop(obs[s]) for s in eachindex(obs)]
    reject = [Int(p <= α) for p in p_value]
    error_rate = mean(reject)

    error_rate
end


# randomization-based MMRM, given a set of observations simulated for one trial
function p_value_mmrm_sara_rnd(obs::Population, ref::Matrix{Int64}; nseq = 15000, seed = 314159)
    # setting up a dataset for analysis    
    obs_ = @pipe DataFrame(obs) |>
        filter(:SARA => x -> !ismissing(x), _) |>
        select(_, :id, :time, :TSO, :SARA, :TRT) |>
        groupby(_, :id) |>
        transform(_, 
            :time => (y -> CategoricalArray([i == 1 ? "BL" : "Visit$(i-1)" for i in eachindex(y)])) => :VISIT,
            :TRT => categorical => :TRTc,
            :id => categorical => :id,
            :SARA => (y -> y .- y[1]) => :CHFBL,
            :SARA => (y -> y[1]) => :SARABL
        )

    # selecting re-randomization sequences at random from the reference set 
    Random.seed!(seed)
    seqids = axes(ref, 2)
    trt = ref[:, sample(seqids, nseq)]

    # performing MMRM fit
    mmrm_formula = @formula(SARA ~ TSO + SARABL * VISIT + (1 | id))    
    mmrm = fit(LinearMixedModel, mmrm_formula, obs_)
        
    # residuals: observed - prediction
    obs_pred = @pipe obs_ |> 
        insertcols(_, :SARA_pred => predict(mmrm)) |> 
        transform(_, [:SARA, :SARA_pred] =>ByRow((obs, pred) -> obs-pred) => :RESIDUAL) |> 
        filter(:VISIT => (value -> value == "Visit4"), _)

    R = obs_pred[!, :RESIDUAL]

    # ranks
    nsbj = length(R)
    a = 1:nsbj
    ā = mean(a)
    
    # statistics based on ranks
    T = obs_pred[!, :TRT]
    Sobs = T[sortperm(R)]' * (a .- ā)
    
    # statistics based on rerandomization
    S = [T' * (a .- ā) for T in eachcol(trt[sortperm(R), :])]

    # p-value
    reject = [Int(abs(S[j]) >= abs(Sobs)) for j in eachindex(S)]
    p_value = mean(reject)

    return p_value
end

function p_value_mmrm_sara_chfbl_rnd(obs::Population, ref::Matrix{Int64}; nseq = 15000, seed = 314159)
    # setting up a dataset for analysis    
    obs_ = @pipe DataFrame(obs) |>
        filter(:SARA => x -> !ismissing(x), _) |>
        select(_, :id, :time, :TSO, :SARA, :TRT) |>
        groupby(_, :id) |>
        transform(_, 
            :time => (y -> CategoricalArray([i == 1 ? "BL" : "Visit$(i-1)" for i in eachindex(y)])) => :VISIT,
            :TRT => categorical => :TRTc,
            :id => categorical => :id,
            :SARA => (y -> y .- y[1]) => :CHFBL,
            :SARA => (y -> y[1]) => :SARABL
        ) |>
        filter(:VISIT => (value -> value != "BL"), _)

    # selecting re-randomization sequences at random from the reference set 
    Random.seed!(seed)
    seqids = axes(ref, 2)
    trt = ref[:, sample(seqids, nseq)]

    # performing MMRM fit
    mmrm_formula = @formula(CHFBL ~ TSO + SARABL * VISIT + (1 | id))
    mmrm = fit(LinearMixedModel, mmrm_formula, obs_)
        
    # residuals: observed - prediction
    obs_pred = @pipe obs_ |> 
        insertcols(_, :CHFBL_pred => predict(mmrm)) |> 
        transform(_, [:CHFBL, :CHFBL_pred] =>ByRow((obs, pred) -> obs-pred) => :RESIDUAL) |> 
        filter(:VISIT => (value -> value == "Visit4"), _)

    R = obs_pred[!, :RESIDUAL]

    # ranks
    nsbj = length(R)
    a = 1:nsbj
    ā = mean(a)
    
    # statistics based on ranks
    T = obs_pred[!, :TRT]
    Sobs = T[sortperm(R)]' * (a .- ā)
    
    # statistics based on rerandomization
    S = [T' * (a .- ā) for T in eachcol(trt[sortperm(R), :])]

    # p-value
    reject = [Int(abs(S[j]) >= abs(Sobs)) for j in eachindex(S)]
    p_value = mean(reject)

    return p_value
end


# randomization-based MMRM, given observations simulated for many trials
function mmrm_sara_rnd(obs::Vector{<:Population}, ref::Matrix{Int64}; α::Number = 0.05, nseq::Int64 = 15000, seed = 314159)
    p_value = [p_value_mmrm_sara_rnd(obs[s], ref, nseq = nseq, seed = seed + 2*s) for s in eachindex(obs)]
    reject = [Int(p .<= α) for p in p_value]
    error_rate = mean(reject)

    return error_rate
end


function mmrm_sara_chfbl_rnd(obs::Vector{<:Population}, ref::Matrix{Int64}; α::Number = 0.05, nseq::Int64 = 15000, seed = 314159)
    p_value = [p_value_mmrm_sara_chfbl_rnd(obs[s], ref, nseq = nseq, seed = seed + 2*s) for s in eachindex(obs)]
    reject = [Int(p .<= α) for p in p_value]
    error_rate = mean(reject)

    return error_rate
end