function fit_sara_logistic_model(obs::Union{Population, DataFrame}, seed::Int64)
    println("seed = $seed")
    # setting initial parameters' values
    Random.seed!(seed)

    # initial paramaters' values
    itvα = rand(Uniform(0.01, 0.21)) # TV of α parameter
    itvβ = rand(Uniform(0.10, 0.70)) # TV of β parameter
    itvγ = 28.75                     # TV of γ parameter
    itvδ = rand(Uniform(3.00, 8.00)) # TV of δ parameter  
     iωα = rand(Uniform(0.05, 0.35)) # BSV for the α parameter
     iωβ = rand(Uniform(0.05, 0.35)) # BSV for the β parameter
     iωγ = 0.0                       # BSV for the γ parameter (by default, γ is fixed)
     iωδ = rand(Uniform(0.05, 0.35)) # BSV for the δ parameter
      iσ = rand(Uniform(0.25, 0.75)) # ISV 
     iDE = rand(Uniform(0.05, 0.95)) # drug effect

    # initial parameters for the Null model  
    null_init_params = (;
        tvα = itvα,  
        tvβ = itvβ,  
        tvγ = itvγ, 
        tvδ = itvδ,  
         ωα = iωα,  
         ωβ = iωβ, 
         ωγ = iωγ, 
         ωδ = iωδ,  
          σ = iσ,
         DE = 0.0    
    )

    # initial parameters for the ALT model  
    alt_init_params = (;
        tvα = itvα,  
        tvβ = itvβ,  
        tvγ = itvγ, 
        tvδ = itvδ,  
         ωα = iωα,  
         ωβ = iωβ, 
         ωγ = iωγ, 
         ωδ = iωδ,  
          σ = iσ,
         DE = iDE    
    )

    # fitting the model, assuming NULL scenario
    null_fit = try 
        fit(
            sara_logistic_model, 
            obs, 
            null_init_params, 
            FOCE(),
            constantcoef=(:tvγ, :ωγ, :DE),
            optim_options = (; show_trace=false) 
        )
        catch e
            nothing
        end
    
    # fitting the model, assuming ALT scenario
    alt_fit = try 
        fit(
            sara_logistic_model, 
            obs, 
            alt_init_params, 
            FOCE(),
            constantcoef=(:tvγ, :ωγ),
            optim_options = (; show_trace=false) 
        )
        catch e
            nothing
        end
        
    if !isnothing(null_fit) & !isnothing(alt_fit)
        return null_fit, alt_fit
    else
        return nothing                
    end
end


# population-based LR test, given a set of observations simulated for one trial
function p_value_nlmem_pop(fitted_models::Tuple)
    # getting fitted models
    null_model, alt_model = fitted_models

    # performing LR test
    test = Pumas.lrtest(null_model, alt_model)
    
    # returning test's p-value
    p_value = Pumas.pvalue(test)

    return p_value   
end


# population-based LR test, given observations simulated for many trials
function nlmem_pop(fitted_models::Vector{<:Tuple}; α::Number = 0.05)
    p_value = [p_value_nlmem_pop(fitted_models[s]) for s in eachindex(fitted_models)]
    reject = [Int(p <= α) for p in p_value]
    error_rate = mean(reject)

    error_rate
end

# function to extract individual objective function values (OFVs) from a fitted Pumas model
function get_individual_ofvs(model_fit::Pumas.FittedPumasModel)
    # Extract the estimated parameters
    final_params = coef(model_fit)

    # Get the approximation method from the fit
    approx_method = model_fit.approx

    # Create an empty array to store individual log-likelihoods
    individual_lls = Float64[]

    # Loop through each subject in the population
    for subject in model_fit.data
        # Calculate the log-likelihood for the subject
        ll = Pumas.loglikelihood(model_fit.model, subject, final_params, approx_method)
        push!(individual_lls, ll)
    end

    # The objective function value is -2 * log-likelihood
    return -2 .* individual_lls
end

# calculates AUC
function auc(t::Vector{<:Number}, y::Vector{<:Number}; method = "trapezoid")
    if method == "trapezoid"
        h = t[2:end] - t[1:end-1]
        return sum(0.5 .* (y[2:end] + y[1:end-1]) .* h)  
    else
        throw(ArgumentError("Incorrect method is provided: must be \"trapezoid\" but \"$method\" is used as an input!"))
    end   
end

# NLMEM randomization-based test, given a fitted model obtained from a set of observations simulated for one trial
function p_value_nlmem_rnd(fitted_models::Tuple, obs::Population, ref::Matrix{Int64}; nseq = 15000, seed = 314159)
    # prediction given H0
    fit_null, _ = fitted_models
    pred_ = [predict(fit_null, obs[i]) for i in eachindex(obs)]

    # treatment assignments
    T = [obs[i].covariates.u[:TRT] for i in eachindex(obs)]
    
    # observed CHFBL
    chfblobs = [obs[i].observations[:SARA][end]-obs[i].observations[:SARA][1] for i in eachindex(obs)]
    
    # predicted CHFBL
    chfblpred = [pred_[i].ipred[:SARA][end]-pred_[i].ipred[:SARA][1] for i in eachindex(pred_)]

    # residuals
    R1 = chfblobs - chfblpred

    # observed AUC
    tobs = [obs[i].time for i in eachindex(obs)]
    yobs = [obs[i].observations[:SARA] for i in eachindex(obs)]
    aucobs = [auc(t, y) for (t, y) in zip(tobs, yobs)]

    # predicted AUC
    tpred = [pred_[i].time for i in eachindex(pred_)]
    ypred = [pred_[i].ipred[:SARA] for i in eachindex(pred_)]
    aucpred = [auc(t, y) for (t, y) in zip(tpred, ypred)]

    # residuals
    R2 = aucobs - aucpred

    # predicted random effects on α
    ηα = [pred_[i].ebes[:ηα] for i in eachindex(pred_)]
    R3 = ηα .- mean(ηα)
    
    # individual OFVs
    iOFV = get_individual_ofvs(fit_null)
    R4 = iOFV .- mean(iOFV)
    
    # selecting re-randomization sequences at random from the reference set 
    Random.seed!(seed)
    seqids = axes(ref, 2)
    trt = ref[:, sample(seqids, nseq)]

    # ranks
    nsbj = length(obs)
    a = 1:nsbj
    ā = mean(a)
    
    # observed statistics (based on ranks)
    S1obs = T[sortperm(R1)]' * (a .- ā)
    S2obs = T[sortperm(R2)]' * (a .- ā)
    S3obs = T[sortperm(R3)]' * (a .- ā)
    S4obs = T[sortperm(R4)]' * (a .- ā)

    # statistics based on rerandomization
    S1 = [T' * (a .- ā) for T in eachcol(trt[sortperm(R1), :])]
    S2 = [T' * (a .- ā) for T in eachcol(trt[sortperm(R2), :])]
    S3 = [T' * (a .- ā) for T in eachcol(trt[sortperm(R3), :])]
    S4 = [T' * (a .- ā) for T in eachcol(trt[sortperm(R4), :])]

    # p-value
    reject1 = [Int(abs(S) >= abs(S1obs)) for S in S1]
    reject2 = [Int(abs(S) >= abs(S2obs)) for S in S2]
    reject3 = [Int(abs(S) >= abs(S3obs)) for S in S3]
    reject4 = [Int(abs(S) >= abs(S4obs)) for S in S4]

    p_value = [mean(reject) for reject in [reject1, reject2, reject3, reject4]]

    return p_value
end


# NLMEM randomization-based test, given a fitted model obtained from a set of observations simulated for many trials
function nlmem_rnd(fitted_models::Vector{<:Tuple}, obs::Vector{<:Population}, ref::Matrix{Int64}; α::Number = 0.05, nseq::Int64 = 15000, seed = 314159)
    p_value = [p_value_nlmem_rnd(fitted_models[s], obs[s], ref, nseq = nseq, seed = seed + 2*s) for s in eachindex(fitted_models) if !isnothing(fitted_models[s])]
    reject = hcat([[Int(p .<= α) for p in pv] for pv in p_value]...)
    error_rate = mean(reject, dims = 2)

    return DataFrame([name => value for (name, value) in zip(["CHFBL", "AUC", "ηα", "iOFV"], error_rate)])
end

