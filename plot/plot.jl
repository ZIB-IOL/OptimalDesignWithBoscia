using CSV
using DataFrames
using Plots
using Plots.PlotMeasures: mm

include(joinpath(@__DIR__, "..", "ODWB", "colours.jl"))
include(joinpath(@__DIR__, "plot_style.jl"))

const TIME_LIMIT = 3600.0
const BOSCIA_DIR = joinpath(@__DIR__, "..", "ODWB", "csv", "Boscia")
const SCIPSDP_DIR = joinpath(@__DIR__, "..", "ODWB", "csv", "SCIPSDP")
const OUT_DIR = joinpath(@__DIR__, "termination_plots")
const REL_TOL = 0.05
const ABS_TOL = 1e-6

rgb_hex(c::NTuple{3, <:Real}) = "#" * join((uppercase(string(v, base=16, pad=2)) for v in round.(Int, 255 .* collect(c))), "")

const CB_PALETTE = [
    rgb_hex(cb_blue),
    rgb_hex(cb_burgundy),
    rgb_hex(cb_green_sea),
    rgb_hex(cb_purple),
    rgb_hex(cb_brown),
    rgb_hex(cb_clay),
]

const SCIPSDP_PALETTE = [
    rgb_hex(cb_green_lime),   # Boscia
    rgb_hex(cb_salmon_pink),  # SCIPSDP OA
    rgb_hex(cb_green_sea),    # SCIPSDP BnB
]

const ABLATION_PALETTE = [
    rgb_hex(cb_green_sea),
    rgb_hex(cb_salmon_pink),
    rgb_hex(cb_clay),
    rgb_hex(cb_blue_light),
    rgb_hex(cb_burgundy),
    rgb_hex(cb_lilac),
]

const PRUNING_PALETTE = [
    rgb_hex(cb_green_sea),
    rgb_hex(cb_salmon_pink),
    rgb_hex(cb_burgundy),
]

const ACST_PRUNING_PALETTE = [
    rgb_hex(cb_green_sea),
    rgb_hex(cb_salmon_pink),
    rgb_hex(cb_burgundy),
    rgb_hex(cb_lilac),
]

Base.@kwdef struct SetupSpec
    label::String
    kind::Symbol
    name::String
    subdir::Union{Nothing, String} = nothing
end

Base.@kwdef struct ExperimentConfig
    criterion::String
    data_type::String
    suffix::String
    file_tag::String
    formulations::Vector{String} = String[]
end

Base.@kwdef struct RunRecord
    time::Float64
    termination::String
    solution::Float64
    path::String
end

const OFFICIAL_KEY_CACHE = Dict{Tuple{String, String}, Set{NTuple{4, Int}}}()

const FAMILY_DEFS = Dict(
    "smoothing_ablation" => [
        SetupSpec(label="baseline", kind=:boscia, name="baseline"),
        SetupSpec(label="scaled μ", kind=:boscia, name="scaled_mu"),
        SetupSpec(label="half spectrum", kind=:boscia, name="reduced_spectrum_2"),
        SetupSpec(label="third spectrum", kind=:boscia, name="reduced_spectrum_3"),
        SetupSpec(label="half spectrum + scaled μ", kind=:boscia, name="reduced_spectrum_2_scaled_mu"),
        SetupSpec(label="third spectrum + scaled μ", kind=:boscia, name="reduced_spectrum_3_scaled_mu"),
    ],
    "exclusion" => [
        SetupSpec(label="baseline", kind=:boscia, name="baseline"),
        SetupSpec(label="dual fixing", kind=:boscia, name="dual_exclusion_criterion"),
        SetupSpec(label="fixings", kind=:boscia, name="exclusion_criterion"),
        SetupSpec(label="fixings random", kind=:boscia, name="exclusion_criterion_random"),
        SetupSpec(label="fixings tighter tol", kind=:boscia, name="exclusion_criterion_tighter_tol"),
    ],
    "pruning" => [
        SetupSpec(label="baseline", kind=:boscia, name="baseline"),
        SetupSpec(label="eigenvalue pruning", kind=:boscia, name="eigenvalue_based_pruning"),
        SetupSpec(label="rank pruning", kind=:boscia, name="rank_based_pruning"),
    ],
    "scipsdp" => [
        SetupSpec(label="Boscia", kind=:boscia, name="baseline"),
        SetupSpec(label="SCIPSDP OA", kind=:scipsdp, name="oa"),
        SetupSpec(label="SCIPSDP BnB", kind=:scipsdp, name="bnb"),
    ],
)

function experiment_configs()
    return [
        ExperimentConfig(criterion="E", data_type="correlated", suffix="correlated__", file_tag="correlated"),
        ExperimentConfig(criterion="E", data_type="independent", suffix="independent__", file_tag="independent"),
        ExperimentConfig(criterion="AGC", data_type="correlated", suffix="correlated_connected", file_tag="correlated"),
        ExperimentConfig(criterion="AGC", data_type="independent", suffix="independent_disconnected", file_tag="independent"),
        ExperimentConfig(criterion="ACST", data_type="independent", suffix="independent__", file_tag="independent", formulations=["ACST", "ACSTS"]),
    ]
end

function scipsdp_boscia_spec(cfg::ExperimentConfig)
    # Match production summary tables: optimized Boscia presets vs SCIPSDP.
    return SetupSpec(label="Boscia", kind=:boscia, name="optimized")
end

function family_specs(family::String, cfg::ExperimentConfig)
    specs = get(FAMILY_DEFS, family, nothing)
    specs === nothing && error("Unknown family $(family)")
    if family == "scipsdp"
        return [scipsdp_boscia_spec(cfg), specs[2], specs[3]]
    end
    return specs
end

function solver_prefix(spec::SetupSpec)
    if spec.kind == :boscia
        return spec.name == "baseline" ? "boscia__" : "boscia_$(spec.name)_"
    elseif spec.kind == :scipsdp
        return "scip_sdp_$(spec.name)_"
    end
    error("Unknown setup kind $(spec.kind)")
end

function candidate_dirs(spec::SetupSpec)
    if spec.kind == :scipsdp
        return [SCIPSDP_DIR]
    end
    dirs = String[]
    if isnothing(spec.subdir)
        push!(dirs, BOSCIA_DIR)
    else
        push!(dirs, joinpath(BOSCIA_DIR, spec.subdir))
        push!(dirs, BOSCIA_DIR)
    end
    return unique(filter(isdir, dirs))
end

function file_regex(spec::SetupSpec, cfg::ExperimentConfig, criterion_name::String=cfg.criterion)
    prefix = solver_prefix(spec)
    separator = endswith(cfg.suffix, "_") ? "" : "_"
    # Some μ-scaled runs record the input scaling between criterion and
    # `_optimality`, e.g. `E_scaled_105.9__optimality_...`.
    optional_scaling = raw"(?:_scaled_[^_]+_)?"
    return Regex("^" * prefix * criterion_name * optional_scaling * "_optimality_" *
                 cfg.suffix * separator * raw"(\d+)_(\d+)_(\d+)_(\d+)\.csv$")
end

function parse_float(value, default=NaN)
    if value === missing || isempty(strip(string(value)))
        return default
    end
    parsed = tryparse(Float64, string(value))
    return isnothing(parsed) ? default : parsed
end

function load_record(path::String)
    df = DataFrame(CSV.File(path))
    nrow(df) == 0 && error("No rows in $(path)")
    row = df[end, :]
    time = parse_float(getproperty(row, :time), TIME_LIMIT)
    termination = string(getproperty(row, :termination))
    solution = if :scaled_solution in propertynames(row)
        parse_float(getproperty(row, :scaled_solution))
    elseif :solution in propertynames(row)
        parse_float(getproperty(row, :solution))
    else
        NaN
    end
    return RunRecord(time=min(time, TIME_LIMIT), termination=termination, solution=solution, path=path)
end

function is_e_allowed(m::Int, n::Int, N::Int)
    m in (170, 200) && return false
    return N != fld(3 * n, 4)
end

function keep_instance(criterion_name::String, key::NTuple{4, Int})
    m, n, N, _ = key
    criterion_name == "E" || return true
    return is_e_allowed(m, n, N)
end

function merged_csv_candidates(criterion_name::String, cfg::ExperimentConfig)
    if criterion_name == "E"
        type = cfg.data_type == "correlated" ? "correlated" : "independent"
        return [
            joinpath(BOSCIA_DIR, "boscia_optimized_E_optimality_$(type)_merged.csv"),
            joinpath(BOSCIA_DIR, "boscia_eigenvalue_based_pruning_E_optimality_$(type)_merged.csv"),
            joinpath(BOSCIA_DIR, "boscia_baseline_E_optimality_$(type)_merged.csv"),
            joinpath(SCIPSDP_DIR, "scip_sdp_oa_E_optimality_$(type)_merged.csv"),
            joinpath(SCIPSDP_DIR, "scip_sdp_bnb_E_optimality_$(type)_merged.csv"),
        ]
    elseif criterion_name == "AGC"
        suffix = cfg.data_type == "correlated" ? "correlated_connected" : "independent_disconnected"
        return [
            joinpath(BOSCIA_DIR, "boscia_optimized_AGC_optimality_$(suffix)_merged.csv"),
            joinpath(BOSCIA_DIR, "boscia_eigenvalue_based_pruning_AGC_optimality_$(suffix)_merged.csv"),
            joinpath(BOSCIA_DIR, "boscia_baseline_AGC_optimality_$(suffix)_merged.csv"),
            joinpath(SCIPSDP_DIR, "scip_sdp_oa_AGC_optimality_$(suffix)_merged.csv"),
            joinpath(SCIPSDP_DIR, "scip_sdp_bnb_AGC_optimality_$(suffix)_merged.csv"),
        ]
    elseif criterion_name == "ACST"
        return [
            joinpath(BOSCIA_DIR, "boscia_optimized_ACST_optimality_independent_merged.csv"),
            joinpath(BOSCIA_DIR, "boscia_baseline_ACST_optimality_independent_merged.csv"),
            joinpath(BOSCIA_DIR, "boscia_rank_based_pruning_ACST_optimality_independent_merged.csv"),
            joinpath(SCIPSDP_DIR, "scip_sdp_oa_ACST_optimality_independent_merged.csv"),
            joinpath(SCIPSDP_DIR, "scip_sdp_bnb_ACST_optimality_independent_merged.csv"),
        ]
    elseif criterion_name == "ACSTS"
        return [
            joinpath(BOSCIA_DIR, "boscia_baseline_ACSTS_optimality_independent_merged.csv"),
            joinpath(BOSCIA_DIR, "boscia_rank_based_pruning_ACSTS_optimality_independent_merged.csv"),
        ]
    end
    return String[]
end

function official_instance_keys(criterion_name::String, cfg::ExperimentConfig)
    cache_key = (criterion_name, cfg.data_type)
    get!(OFFICIAL_KEY_CACHE, cache_key) do
        keys = Set{NTuple{4, Int}}()
        for path in merged_csv_candidates(criterion_name, cfg)
            isfile(path) || continue
            for row in CSV.File(path)
                key = (
                    Int(row.numberOfExperiments),
                    Int(row.numberOfParameters),
                    Int(row.N),
                    Int(row.seed),
                )
                push!(keys, key)
            end
        end
        keys
    end
end

function keep_plot_instance(cfg::ExperimentConfig, criterion_name::String, key::NTuple{4, Int})
    keep_instance(criterion_name, key) || return false
    if criterion_name in ("E", "AGC", "ACST", "ACSTS")
        official_keys = official_instance_keys(criterion_name, cfg)
        return isempty(official_keys) || (key in official_keys)
    end
    return true
end

termination_is_optimal(termination::AbstractString) = strip(termination) == "OPTIMAL"

function is_solved(record::RunRecord)
    return termination_is_optimal(record.termination) && isfinite(record.time) && record.time < TIME_LIMIT
end

function collect_runs(spec::SetupSpec, cfg::ExperimentConfig, criterion_name::String=cfg.criterion)
    regex = file_regex(spec, cfg, criterion_name)
    runs = Dict{NTuple{4, Int}, RunRecord}()
    for dir in candidate_dirs(spec)
        for fname in readdir(dir)
            match_obj = match(regex, fname)
            isnothing(match_obj) && continue
            key = Tuple(parse(Int, part) for part in match_obj.captures)::NTuple{4, Int}
            keep_plot_instance(cfg, criterion_name, key) || continue
            haskey(runs, key) && continue
            runs[key] = load_record(joinpath(dir, fname))
        end
    end
    return runs
end

function merged_csv_path_for_spec(spec::SetupSpec, cfg::ExperimentConfig, criterion_name::String)
    if spec.kind == :scipsdp
        if criterion_name == "E"
            path_cont = joinpath(SCIPSDP_DIR, "scip_sdp_$(spec.name)_E_optimality_$(cfg.data_type)_cont_merged.csv")
            path_plain = joinpath(SCIPSDP_DIR, "scip_sdp_$(spec.name)_E_optimality_$(cfg.data_type)_merged.csv")
            path = isfile(path_cont) ? path_cont : path_plain
            return isfile(path) ? path : nothing
        end
        suffix = if criterion_name == "AGC"
            cfg.data_type == "correlated" ? "correlated_connected" : "independent_disconnected"
        elseif criterion_name == "ACST"
            "independent"
        else
            return nothing
        end
        path = joinpath(SCIPSDP_DIR, "scip_sdp_$(spec.name)_$(criterion_name)_optimality_$(suffix)_merged.csv")
        return isfile(path) ? path : nothing
    end

    if spec.kind == :boscia
        suffix = if criterion_name == "E"
            cfg.data_type
        elseif criterion_name == "AGC"
            cfg.data_type == "correlated" ? "correlated_connected" : "independent_disconnected"
        elseif criterion_name in ("ACST", "ACSTS")
            "independent"
        else
            return nothing
        end
        folder = spec.name == "baseline" ? "baseline" : spec.name
        path = joinpath(BOSCIA_DIR, "boscia_$(folder)_$(criterion_name)_optimality_$(suffix)_merged.csv")
        return isfile(path) ? path : nothing
    end
    return nothing
end

function collect_runs_from_merged(spec::SetupSpec, cfg::ExperimentConfig, criterion_name::String=cfg.criterion)
    path = merged_csv_path_for_spec(spec, cfg, criterion_name)
    isnothing(path) && return collect_runs(spec, cfg, criterion_name)
    delim = spec.kind == :scipsdp ? ',' : ';'
    runs = Dict{NTuple{4, Int}, RunRecord}()
    for row in CSV.File(path; delim=delim)
        key = (
            Int(row.numberOfExperiments),
            Int(row.numberOfParameters),
            Int(row.N),
            Int(row.seed),
        )::NTuple{4, Int}
        keep_plot_instance(cfg, criterion_name, key) || continue
        haskey(runs, key) && continue
        termination = string(row.termination)
        time = parse_float(getproperty(row, :time), TIME_LIMIT)
        solution = hasproperty(row, :scaled_solution) ? parse_float(getproperty(row, :scaled_solution)) : NaN
        runs[key] = RunRecord(time=min(time, TIME_LIMIT), termination=termination, solution=solution, path=path)
    end
    return runs
end

function quasioptimal_keys(run_maps::Dict{String, Dict{NTuple{4, Int}, RunRecord}}; sense::Symbol=:min)
    sense in (:min, :max) || error("sense must be :min or :max")
    labels = collect(keys(run_maps))
    all_keys = Set{NTuple{4, Int}}()
    for runs in values(run_maps), key in keys(runs)
        push!(all_keys, key)
    end

    flagged = Dict(label => Set{NTuple{4, Int}}() for label in labels)
    for key in all_keys
        if sense == :min
            best = Inf
            for label in labels
                record = get(run_maps[label], key, nothing)
                record === nothing && continue
                isfinite(record.solution) || continue
                best = min(best, record.solution)
            end
            isfinite(best) || continue
            tol = max(REL_TOL * abs(best), ABS_TOL)
            for label in labels
                record = get(run_maps[label], key, nothing)
                record === nothing && continue
                termination_is_optimal(record.termination) || continue
                isfinite(record.solution) || continue
                if record.solution > best + tol
                    push!(flagged[label], key)
                end
            end
        else
            best = -Inf
            for label in labels
                record = get(run_maps[label], key, nothing)
                record === nothing && continue
                isfinite(record.solution) || continue
                best = max(best, record.solution)
            end
            isfinite(best) || continue
            tol = max(REL_TOL * abs(best), ABS_TOL)
            for label in labels
                record = get(run_maps[label], key, nothing)
                record === nothing && continue
                termination_is_optimal(record.termination) || continue
                isfinite(record.solution) || continue
                if record.solution < best - tol
                    push!(flagged[label], key)
                end
            end
        end
    end
    return flagged
end

function solved_times(runs::Dict{NTuple{4, Int}, RunRecord}, flagged::Set{NTuple{4, Int}})
    times = Float64[]
    for (key, record) in runs
        key in flagged && continue
        is_solved(record) || continue
        push!(times, min(record.time, TIME_LIMIT))
    end
    sort!(times)
    return times
end

function fixed_reference_quasioptimal_keys(run_maps::Dict{String, Dict{NTuple{4, Int}, RunRecord}}, reference_label::String)
    labels = collect(keys(run_maps))
    reference_runs = get(run_maps, reference_label, Dict{NTuple{4, Int}, RunRecord}())
    flagged = Dict(label => Set{NTuple{4, Int}}() for label in labels)
    isempty(reference_runs) && return flagged
    for (key, ref_record) in reference_runs
        isfinite(ref_record.solution) || continue
        tol = max(REL_TOL * abs(ref_record.solution), ABS_TOL)
        for label in labels
            label == reference_label && continue
            record = get(run_maps[label], key, nothing)
            record === nothing && continue
            termination_is_optimal(record.termination) || continue
            isfinite(record.solution) || continue
            if record.solution > ref_record.solution + tol
                push!(flagged[label], key)
            end
        end
    end
    return flagged
end

function step_xy(times::Vector{Float64})
    if isempty(times)
        return [1.0, TIME_LIMIT], [0, 0]
    end
    x = Float64[1.0]
    y = Int[0]
    solved = 0
    for t in times
        push!(x, t)
        push!(y, solved)
        solved += 1
        push!(x, t)
        push!(y, solved)
    end
    push!(x, TIME_LIMIT)
    push!(y, solved)
    return x, y
end

function plot_path(family::String, cfg::ExperimentConfig)
    mkpath(OUT_DIR)
    return joinpath(OUT_DIR, "$(cfg.criterion)_$(cfg.file_tag)_$(family).pdf")
end

function style_triplets(n::Int; colors=CB_PALETTE)
    markers = [:circle, :rect, :utriangle, :diamond, :pentagon, :xcross]
    linestyles = [:solid, :dash, :dashdot, :dot, :solid, :dash]
    return [
        (
            colors[mod1(i, length(colors))],
            markers[mod1(i, length(markers))],
            linestyles[mod1(i, length(linestyles))],
        )
        for i in 1:n
    ]
end

function total_instance_count(run_maps::Dict{String, Dict{NTuple{4, Int}, RunRecord}}, labels::Vector{String})
    instance_keys = Set{NTuple{4, Int}}()
    for label in labels, key in keys(run_maps[label])
        push!(instance_keys, key)
    end
    return length(instance_keys)
end

function formulations_for_family(cfg::ExperimentConfig, family::String)
    if isempty(cfg.formulations)
        return [cfg.criterion]
    end
    family == "scipsdp" && return [cfg.criterion]
    return cfg.formulations
end

function generate_plot(family::String, cfg::ExperimentConfig; verbose::Bool=true)
    specs = family_specs(family, cfg)

    formulations = formulations_for_family(cfg, family)
    use_formulation_suffix = length(formulations) > 1
    run_maps = Dict{String, Dict{NTuple{4, Int}, RunRecord}}()
    series_entries = Tuple{String, String}[]
    for criterion_name in formulations
        formulation_suffix = use_formulation_suffix ? " ($(criterion_name))" : ""
        for spec in specs
            series_label = spec.label * formulation_suffix
            run_maps[series_label] = family == "scipsdp" ? collect_runs_from_merged(spec, cfg, criterion_name) : collect_runs(spec, cfg, criterion_name)
            push!(series_entries, (series_label, spec.label))
        end
    end
    present_labels = [series_label for (series_label, _) in series_entries if !isempty(run_maps[series_label])]
    if length(present_labels) < 2
        verbose && println("Skipping $(family) / $(cfg.criterion) / $(cfg.data_type): fewer than two setups with runs.")
        return false
    end

    sense = cfg.criterion == "ACST" ? :max : :min
    flagged = if family == "scipsdp" && cfg.criterion == "ACST"
        fixed_reference_quasioptimal_keys(run_maps, "Boscia")
    else
        quasioptimal_keys(run_maps; sense)
    end
    total_instances = total_instance_count(run_maps, present_labels)
    plt = plot(
        xscale=:log10,
        xlim=(1.0, TIME_LIMIT),
        ylim=(0, max(total_instances, 1)),
        xlabel="Time (s)",
        ylabel="Solved instances",
        grid=true,
        legend=:topleft,
        size=(820, 500);
        cm_plot_kwargs(guidefontsize=13, tickfontsize=11, legendfontsize=11)...
    )

    colors = if family == "scipsdp"
        SCIPSDP_PALETTE
    elseif family == "pruning"
        PRUNING_PALETTE
    else
        ABLATION_PALETTE
    end
    base_styles = Dict(
        spec.label => style
        for (spec, style) in zip(specs, style_triplets(length(specs); colors))
    )
    series_styles = if family == "pruning" && use_formulation_suffix
        visible_entries = [
            (series_label, base_label)
            for (series_label, base_label) in series_entries
            if !isempty(run_maps[series_label])
        ]
        Dict(
            series_label => style
            for ((series_label, _), style) in zip(
                visible_entries,
                style_triplets(length(visible_entries); colors=ACST_PRUNING_PALETTE),
            )
        )
    else
        Dict{String, Tuple{String, Symbol, Symbol}}()
    end
    for (series_label, base_label) in series_entries
        runs = run_maps[series_label]
        isempty(runs) && continue
        times = solved_times(runs, flagged[series_label])
        x, y = step_xy(times)
        color, marker, linestyle = get(series_styles, series_label, base_styles[base_label])
        if use_formulation_suffix && endswith(series_label, "(ACSTS)")
            linestyle = :dash
        end
        plot!(
            plt,
            x,
            y,
            label=series_label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=4,
        )
    end

    out = plot_path(family, cfg)
    savefig(plt, out)
    verbose && println("Wrote $(out)")
    return true
end

function parse_arg_list(args::Vector{String}, flag::String, default::Vector{String})
    idx = findfirst(==(flag), args)
    isnothing(idx) && return default
    idx == length(args) && error("Expected comma-separated value after $(flag)")
    return String[strip(part) for part in split(args[idx + 1], ",") if !isempty(strip(part))]
end

function selected_configs(criteria::AbstractVector{<:AbstractString}, data_types::AbstractVector{<:AbstractString})
    return [
        cfg for cfg in experiment_configs()
        if cfg.criterion in criteria && cfg.data_type in data_types
    ]
end

function main(args=ARGS)
    families = parse_arg_list(args, "--families", collect(keys(FAMILY_DEFS)))
    criteria = parse_arg_list(args, "--criteria", unique([cfg.criterion for cfg in experiment_configs()]))
    data_types = parse_arg_list(args, "--data-types", unique([cfg.data_type for cfg in experiment_configs()]))

    configs = selected_configs(criteria, data_types)
    isempty(configs) && error("No experiment configurations selected.")

    generated = 0
    for family in families
        haskey(FAMILY_DEFS, family) || error("Unknown family $(family)")
        for cfg in configs
            generated += generate_plot(family, cfg) ? 1 : 0
        end
    end

    println("Generated $(generated) termination plots in $(OUT_DIR)")
end

main()
