using CategoricalArrays
using DataFrames
using Dates
using Distributions
using GLM
using HypothesisTests
using JLD2
using MixedModels
using Pumas
using PumasUtilities
using Random
using Pipe

include("00-utils.jl")
include("01-simulate-data.jl")
include("02-randomization.jl")
include("03-t-test.jl")
include("04-ancova.jl")
include("05-mmrm.jl")
include("06-nlmem.jl")

# ===== Simulation setup =====
# sample size
nsbj = 100

# seed
seed = 314159

# number of simulations
nsim = 2500 #5000

# Drug effect
DE = Dict(
    :NULL => 0.00, 
    :ALT1 => 0.50, 
    :ALT2 => 0.75,
    :ALT3 => 1.00
)

# time trend
trend = Dict(
    :notrend => false,
    :trend => true
)

# randomization sequences
trt = Dict(
    :RND => [Rand(nsbj, seed + 2*s) for s in 1:nsim],
    :TBD => [tbd(nsbj, seed + 2*s)  for s in 1:nsim]
);

# re-randomization sequences (for randomization-based inference)
nseqmax = 1_000_000
@time reftrt = Dict(
    :RND => hcat([Rand(nsbj, seed + 2*s) for s in 1:nseqmax]...),
    :TBD => hcat([tbd(nsbj, seed + 2*s)  for s in 1:nseqmax]...)
);

# ===== Simulating observations =====
obs = Dict{Symbol, Dict}()
for sc in eachindex(DE)
    println("Simulating $sc scenario ...")
    setindex!(obs, Dict{Symbol, Dict}(), sc)
    for tt in eachindex(trend)
        println(" - time trend: $tt")
        setindex!(obs[sc], Dict{Symbol, Vector{<:Population}}(), tt)
        for rnd in eachindex(trt)
            println("    + radnomization: $rnd")
            simulated_obs = [simulate_obs(nsbj, trt[rnd][s], trend[tt], DE = DE[sc], seed = seed + 2*s) for s in 1:nsim]
            setindex!(obs[sc][tt], simulated_obs, rnd)
        end
    end
end


# ===== Fitting NLMEM =====
fits = Dict{Symbol, Dict}()
for sc in eachindex(DE)
    println("Fitting models for $sc scenario ...")
    setindex!(fits, Dict{Symbol, Dict}(), sc)
    for tt in eachindex(trend)
        println(" - time trend: $tt")
        setindex!(fits[sc], Dict{Symbol, Union{Nothing, Vector{<:Tuple}}}(), tt)
        for rnd in eachindex(trt)
            println("    + radnomization: $rnd")

            # setting up a file name to save fitted models
            model_fits_file = "model_fits01/fit$(sc)_$(tt)_$(rnd).jld2"

            # if file exists, ther read the fitted models and add to the dictionary
            if isfile(model_fits_file)
                println("reading " * model_fits_file * " ...")
                model_fits = load_object(model_fits_file)
                setindex!(fits[sc][tt], model_fits, rnd)
            else
                seed_ = [s for s in 1:nsim]
                if (sc == :ALT1) & (tt == :trend)
                    seed_ = rnd == :RND ? [s in [2046] ? 2*s : s for s in 1:nsim] : [s in [456, 905] ? 2*s : s for s in 1:nsim]
                end
                if (sc == :ALT2) & (tt == :notrend)
                    seed_ = rnd == :RND ? [s in [768, 1031] ? 2*s : s for s in 1:nsim] : [s in [456, 1470, 2224] ? 2*s : s for s in 1:nsim]
                end
                if (sc == :ALT2) & (tt == :trend)
                    seed_ = rnd == :RND ? [s in [1903] ? 2*s : s for s in 1:nsim] : [s for s in 1:nsim]
                end
                if (sc == :ALT3) & (tt == :notrend)
                    seed_ = rnd == :RND ? [s in [112] ? 2*s : s for s in 1:nsim] : [s in [1529, 2438] ? 2*s : s for s in 1:nsim]
                end
                if (sc == :ALT3) & (tt == :trend)
                    seed_ = rnd == :RND ? [s for s in 1:nsim] : [s in [1484] ? 2*s : s for s in 1:nsim]
                end
                if (sc == :NULL) & (tt == :notrend)
                    seed_ = rnd == :RND ? [s for s in 1:nsim] : [s in [1528] ? 2*s : s for s in 1:nsim]
                end
                if (sc == :NULL) & (tt == :trend)
                    seed_ = rnd == :RND ? [s in [327] ? 2*s : s for s in 1:nsim] : [s in [327] ? 2*s : s for s in 1:nsim]
                end

                model_fits = [fit_sara_logistic_model(obs[sc][tt][rnd][s], seed_[s]) for s in 1:nsim];
                model_fits = [model_fits[s] for s in eachindex(model_fits) if !isnothing(model_fits[s])]
                setindex!(fits[sc][tt], model_fits, rnd)

                # saving fitted models
                save_object(model_fits_file, model_fits)
            end    
        end
    end
end


## ================ Inference: ================
## LR-test for NLMEM
output_pop_names = ["Case", "Treatment effect", "Linear trend", "Randomization", "Population-based"]
output_rnd_names = ["Case", "Treatment effect", "Linear trend", "Randomization", "CHFBL", "AUC", "ηα", "iOFV"]

output_pop_types = [String, Float64, String, String, Float64]
output_rnd_types = [String, Float64, String, String, Float64, Float64, Float64, Float64]

nlmem_pop_output = DataFrame([name => T[] for (name, T) in zip(output_pop_names, output_pop_types)])
nlmem_rnd_output = DataFrame([name => T[] for (name, T) in zip(output_rnd_names, output_rnd_types)])
for sc in eachindex(fits)
    for tt in eachindex(fits[sc])
        for rnd in eachindex(fits[sc][tt])
            println("Inferencing $sc scenario with $tt ($rnd randomization) ...")
                
            println(" - population-based inference ...")
            push!(nlmem_pop_output, [String(sc), DE[sc], String(tt), String(rnd), nlmem_pop(fits[sc][tt][rnd])])

            println(" - randomization-based inference ...")
            prnd = nlmem_rnd(fits[sc][tt][rnd], obs[sc][tt][rnd], reftrt[rnd])
            push!(nlmem_rnd_output, [String(sc), DE[sc], String(tt), String(rnd), prnd[1, :CHFBL], prnd[1, :AUC], prnd[1, :ηα], prnd[1, :iOFV]])
        end
    end
end

nlmem_pop_output_ = @pipe nlmem_pop_output |>
    transform(_, "Case" => (x -> categorical(x, ordered=true, levels=["NULL", "ALT1", "ALT2", "ALT3"])) => "Case") |>
    transform(_, "Linear trend" => (x -> recode(x, "trend" => "Yes", "notrend" => "No")) => "Linear trend") |>
    transform(_, "Linear trend" => (x -> categorical(x, ordered=true, levels=["Yes", "No"])) => "Linear trend") |>
    transform(_, "Randomization" => (x -> categorical(x, ordered=true, levels=["RND", "TBD"])) => "Randomization") |>
    sort(_, ["Case", "Linear trend", "Randomization"]) |>
    select(_, Not("Case")) |>
    insertcols(_, 1, "scenario" => 1:nrow(_)) |>
    transform(_, ["scenario", "Linear trend"] => ByRow((i, s) -> i % 2 == 0 ? "" : s) => "Linear trend") |>
    transform(_, ["scenario", "Treatment effect"] => ByRow((i, s) -> i % 4 == 1 ? s : " ") => "Treatment effect") |>
    transform(_, "Randomization" => ByRow(x -> x == "RND" ? "Rand" : x) => "Randomization")

nlmem_rnd_output_ = @pipe nlmem_rnd_output |>
    transform(_, "Case" => (x -> categorical(x, ordered=true, levels=["NULL", "ALT1", "ALT2", "ALT3"])) => "Case") |>
    transform(_, "Linear trend" => (x -> recode(x, "trend" => "Yes", "notrend" => "No")) => "Linear trend") |>
    transform(_, "Linear trend" => (x -> categorical(x, ordered=true, levels=["Yes", "No"])) => "Linear trend") |>
    transform(_, "Randomization" => (x -> categorical(x, ordered=true, levels=["RND", "TBD"])) => "Randomization") |>
    sort(_, ["Case", "Linear trend", "Randomization"]) |>
    select(_, Not("Case")) |>
    insertcols(_, 1, "scenario" => 1:nrow(_)) |>
    transform(_, ["scenario", "Linear trend"] => ByRow((i, s) -> i % 2 == 0 ? "" : s) => "Linear trend") |>
    transform(_, ["scenario", "Treatment effect"] => ByRow((i, s) -> i % 4 == 1 ? s : " ") => "Treatment effect") |>
    transform(_, "Randomization" => ByRow(x -> x == "RND" ? "Rand" : x) => "Randomization")

## combining outputs   
nlmem_output = @pipe innerjoin(
    nlmem_pop_output_, 
    nlmem_rnd_output_,
    on = ["scenario", "Treatment effect", "Linear trend", "Randomization"]
) |>
    transform(_, "Treatment effect" => ByRow(x -> convert(Union{Float64, String}, x)) => "Treatment effect") |>
    transform(_, "Linear trend" => ByRow(x -> convert(String, x)) => "Linear trend") |>
    transform(_, "Randomization" => ByRow(x -> convert(String, x)) => "Randomization") 
    

## Inference: t-test, ANCOVA, MMRM
function summarize(obs::Dict{Symbol, Dict}, method_pop::Function, method_rnd::Function, reftrt::Dict{Symbol, Matrix{Int64}} = reftrt)
    output_pop_names = ["Case", "Treatment effect", "Linear trend", "Randomization", "Population-based"]
    output_rnd_names = ["Case", "Treatment effect", "Linear trend", "Randomization", "Randomization-based"]
    output_types = [String, Float64, String, String, Float64]
    
    pop_output = DataFrame([name => T[] for (name, T) in zip(output_pop_names, output_types)])
    rnd_output = DataFrame([name => T[] for (name, T) in zip(output_rnd_names, output_types)])
    for sc in eachindex(obs)
        for tt in eachindex(obs[sc])
            for rnd in eachindex(obs[sc][tt])
                println("Inferencing $sc scenario with $tt ($rnd randomization) ...")
                
                println(" - population-based inference ...") 
                push!(pop_output, [String(sc), DE[sc], String(tt), String(rnd), method_pop(obs[sc][tt][rnd])])
                
                println(" - randomization-based inference ...")
                push!(rnd_output, [String(sc), DE[sc], String(tt), String(rnd), method_rnd(obs[sc][tt][rnd], reftrt[rnd])])
            end
        end
    end

    # formatting output (population-based inference)
    pop_output_ = @pipe pop_output |>
        transform(_, "Case" => (x -> categorical(x, ordered=true, levels=["NULL", "ALT1", "ALT2", "ALT3"])) => "Case") |>
        transform(_, "Linear trend" => (x -> recode(x, "trend" => "Yes", "notrend" => "No")) => "Linear trend") |>
        transform(_, "Linear trend" => (x -> categorical(x, ordered=true, levels=["Yes", "No"])) => "Linear trend") |>
        transform(_, "Randomization" => (x -> categorical(x, ordered=true, levels=["RND", "TBD"])) => "Randomization") |>
        sort(_, ["Case", "Linear trend", "Randomization"]) |>
        select(_, Not("Case")) |>
        insertcols(_, 1, "scenario" => 1:nrow(_)) |>
        transform(_, ["scenario", "Linear trend"] => ByRow((i, s) -> i % 2 == 0 ? "" : s) => "Linear trend") |>
        transform(_, ["scenario", "Treatment effect"] => ByRow((i, s) -> i % 4 == 1 ? s : " ") => "Treatment effect") |>
        transform(_, "Randomization" => ByRow(x -> x == "RND" ? "Rand" : x) => "Randomization")

    # formatting output (randomization-based inference)
    rnd_output_ = @pipe rnd_output |>
        transform(_, "Case" => (x -> categorical(x, ordered=true, levels=["NULL", "ALT1", "ALT2", "ALT3"])) => "Case") |>
        transform(_, "Linear trend" => (x -> recode(x, "trend" => "Yes", "notrend" => "No")) => "Linear trend") |>
        transform(_, "Linear trend" => (x -> categorical(x, ordered=true, levels=["Yes", "No"])) => "Linear trend") |>
        transform(_, "Randomization" => (x -> categorical(x, ordered=true, levels=["RND", "TBD"])) => "Randomization") |>
        sort(_, ["Case", "Linear trend", "Randomization"]) |>
        select(_, Not("Case")) |>
        insertcols(_, 1, "scenario" => 1:nrow(_)) |>
        transform(_, ["scenario", "Linear trend"] => ByRow((i, s) -> i % 2 == 0 ? "" : s) => "Linear trend") |>
        transform(_, ["scenario", "Treatment effect"] => ByRow((i, s) -> i % 4 == 1 ? s : " ") => "Treatment effect") |>
        transform(_, "Randomization" => ByRow(x -> x == "RND" ? "Rand" : x) => "Randomization")

    
    ## combining outputs
    output = @pipe innerjoin(
        pop_output_, 
        rnd_output_,
        on = ["scenario", "Treatment effect", "Linear trend", "Randomization"]
    ) |>
    transform(_, "Treatment effect" => ByRow(x -> convert(Union{Float64, String}, x)) => "Treatment effect") |>
    transform(_, "Linear trend" => ByRow(x -> convert(String, x)) => "Linear trend") |>
    transform(_, "Randomization" => ByRow(x -> convert(String, x)) => "Randomization")

    return output
end

t_test_output           = summarize(obs, t_test_pop, t_test_rnd)
ancova_output           = summarize(obs, ancova_pop, ancova_rnd)
mmrm_sara_output        = summarize(obs, mmrm_sara_pop, mmrm_sara_rnd)
mmrm_sara_chfbl_output  = summarize(obs, mmrm_sara_chfbl_pop, mmrm_sara_chfbl_rnd)

# getting current date for file names
current_date = Dates.format(now(), "yyyymmdd")

# setting up file names
t_test_file          = current_date * "-table01a-t-test.xlsx" 
ancova_file          = current_date * "-table02a-ancova.xlsx" 
mmrm_sara_file       = current_date * "-table03a-mmrm.xlsx" 
mmrm_sara_chfbl_file = current_date * "-table03b-mmrm.xlsx" 
nlmem_file           = current_date * "-table04a-nlmem.xlsx"

# saving outputs
XLSX.writetable(t_test_file, t_test_output, overwrite = true) 
XLSX.writetable(ancova_file, ancova_output, overwrite = true)
XLSX.writetable(mmrm_sara_file, mmrm_sara_output, overwrite = true)
XLSX.writetable(mmrm_sara_chfbl_file, mmrm_sara_chfbl_output, overwrite = true)
XLSX.writetable(nlmem_file, nlmem_output, overwrite = true) 


## power correction for inflated cases: 2STT (trend, TBD), MMRM (trend/notrend, RND/TBD)
function correct_t_test_power(scenario::Symbol, trend::Symbol, rnd::Symbol, obs::Dict{Symbol, Dict})
    # test statistics under the null hypothesis
    Z0 = zeros(Float64, length(obs[:NULL][trend][rnd]))

    # test statistics under the alternative hypothesis
    Z1 = zeros(Float64, length(obs[scenario][trend][rnd]))

    for s in eachindex(Z0)
        println("simulation #$s")
        obs_null = @pipe DataFrame(obs[:NULL][trend][rnd][s]) |>
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
            filter(:VISIT => (value -> value == "Visit4"), _)
        T0 = Int.(obs_null[!, :TRT])
        Y0 = float(obs_null[!, :CHFBL])
        test_null = EqualVarianceTTest(Y0[T0 .== 1], Y0[T0 .== 0])

        obs_alt = @pipe DataFrame(obs[scenario][trend][rnd][s]) |>
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
            filter(:VISIT => (value -> value == "Visit4"), _)
        TA = Int.(obs_alt[!, :TRT])
        YA = float(obs_alt[!, :CHFBL])
        test_alt = EqualVarianceTTest(YA[TA .== 1], YA[TA .== 0])

        # getting test statistics
        Z0[s] = test_null.xbar / test_null.stderr 
        Z1[s] = test_alt.xbar  / test_alt.stderr  
    end
    # quantiles for the null hypothesis
    q0 = quantile(Z0, [0.025, 0.975])

    # corrected power for the alternative hypothesis
    reject = [Int(Z1[s] <= q0[1] || Z1[s] >= q0[2]) for s in eachindex(Z1)] 
        
    return mean(reject)
end


function correct_mmrm_power(scenario::Symbol, trend::Symbol, rnd::Symbol, obs::Dict{Symbol, Dict})
    # test statistics under the null hypothesis
    Z0 = zeros(Float64, length(obs[:NULL][trend][rnd]))

    # test statistics under the alternative hypothesis
    Z1 = zeros(Float64, length(obs[scenario][trend][rnd]))

    for s in eachindex(Z0)
        println("simulation #$s")
        obs_null = @pipe DataFrame(obs[:NULL][trend][rnd][s]) |>
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

        obs_alt = @pipe DataFrame(obs[scenario][trend][rnd][s]) |>
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

        mmrm_formula = @formula(SARA ~ TSO + (SARABL + TRTc) * VISIT + (1 | id))  
        mmrm_null = fit(LinearMixedModel, mmrm_formula, obs_null)
        mmrm_alt = fit(LinearMixedModel, mmrm_formula, obs_alt)

        # getting test statistics
        Z0[s] = mmrm_null.beta[end] ./ mmrm_null.stderror[end] 
        Z1[s] = mmrm_alt.beta[end]  ./ mmrm_alt.stderror[end]  
    end
    # quantiles for the null hypothesis
    q0 = quantile(Z0, [0.025, 0.975])

    # corrected power for the alternative hypothesis
    reject = [Int(Z1[s] <= q0[1] || Z1[s] >= q0[2]) for s in eachindex(Z1)] 
        
    return mean(reject)
end

correct_t_test_power(:ALT1, :trend, :TBD, obs)
correct_mmrm_power(:ALT1, :trend, :RND, obs)
correct_mmrm_power(:ALT1, :trend, :TBD, obs)
correct_mmrm_power(:ALT1, :notrend, :RND, obs)
correct_mmrm_power(:ALT1, :notrend, :TBD, obs)