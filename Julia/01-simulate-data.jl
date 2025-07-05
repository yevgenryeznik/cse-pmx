# logistic NLMEM for the SARA score
sara_logistic_model = @model begin
    @param begin
        # typical values of the model parameters
        tvα ∈ RealDomain(; lower = 0)
        tvβ ∈ RealDomain(; lower = 0)
        tvγ ∈ RealDomain(; lower = 0)
        tvδ ∈ RealDomain(; lower = 0)

        # BSV parameters
        ωα ∈ RealDomain(; lower = 0)
        ωβ ∈ RealDomain(; lower = 0)
        ωγ ∈ RealDomain(; lower = 0)
        ωδ ∈ RealDomain(; lower = 0)

        # measurement error
        σ ∈ RealDomain(; lower = 0.00001)

        # drug effect
        DE ∈ RealDomain(; lower = 0, upper = 1)
    end

    @random begin
        # defining random effects
        ηα ~ Normal(0, √ωα)
        ηβ ~ Normal(0, √ωβ)
        ηγ ~ Normal(0, √ωγ)
        ηδ ~ Normal(0, √ωδ)
    end

    @covariates begin
        # time since onset at BL
        TSO
        TRT
    end

    @pre begin
        # generating individual parameters
        α = tvα * exp(ηα)
        β = tvβ * exp(ηβ)
        γ = tvγ * exp(ηγ)
        δ = tvδ * exp(ηδ) 
        EFF = TRT == 1 ? (1 - DE) : 1
    end

    @derived begin
        Y := @. δ + γ / (1 + exp(β - α/12 * TSO - α/12 * EFF * (t - TSO)))
        SARA ~ @. Normal(Y, σ)
    end
end


function simulate_obs(
    nsbj::Int64,           # sample size
    trt::Vector{Int64},    # vector of tretament assignments
    trend::Bool;           # presence of time trend
    years::Int64  =  2,    # study duration (in years)    
      tvα::Number =  0.11, # TV of α parameter
      tvβ::Number =  3.94, # TV of β parameter
      tvγ::Number = 28.75, # TV of γ parameter
      tvδ::Number =  6.16, # TV of δ parameter  
       ωα::Number =  0.09, # BSV for the α parameter
       ωβ::Number =  0.20, # BSV for the β parameter
       ωγ::Number =  0.00, # BSV for the γ parameter (by default, γ is fixed)
       ωδ::Number =  0.31, # BSV for the δ parameter
        σ::Number =  0.5,  # measurement error variability (std)
       DE::Number =  0.0,  # drug effect (by default, no drug effect)
     seed::Int64  = 314159 # a seed for generating random observations
)
    # collecting population parameters
    model_parameters = (; 
        tvα = tvα, tvβ = tvβ, tvγ = tvγ, tvδ = tvδ,  
        ωα = ωα, ωβ = ωβ, ωγ = ωγ, ωδ = ωδ,
        σ =  σ, DE = DE
    )

    # generating time since onset (TSO) at baseline (BL)
    Random.seed!(seed)
    TSO = sample(0:360, nsbj, replace=true)
    if trend
        sort!(TSO, rev = true)
    end

    # categorizing onset, give TSO values
    onset = map(TSO) do tso
        if (0 <= tso < 120)
            return "Early"
        elseif (120 <= tso < 240)
            return "Moderate"
        else
            return "Late"
        end
    end

    # subject IDs
    sbj = collect(1:nsbj)

    # initializing population:
    pop = map(i -> Subject(; id = i, covariates = (; TSO = TSO[i], TRT = trt[i], onset = onset[i])), sbj)

    # simulating observations:
    obs = [simobs(
        sara_logistic_model, 
        pop[i], 
        model_parameters; 
        obstimes = [TSO[i] + k for k in collect(0:6:(years*12))]
    ) for i in eachindex(sbj)]
    obs = [Subject(obs[i]) for i in eachindex(obs)]

    return obs
end