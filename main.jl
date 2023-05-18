using MultipleScattering
using EffectiveWaves
using Plots
using DataFrames, CSV
using ProgressMeter
using LaTeXStrings
using Dates

include("averaged_multipole_decomposition.jl")
include("common_params.jl")
include("EffectiveSphere.jl")
include("MonteCarloResult.jl")
include("plotMC.jl")



Ω = collect(0.05:0.015:.5);
basis_order = 10;
basis_field_order = 4;

kws_MC = Dict(
    :radius_big_cylinder=>radius_big_cylinder
    ,:basis_order=> basis_order
    ,:basis_field_order=> basis_field_order
    ,:nb_iterations_max=> 5000
    ,:nb_iterations_step=> 200
    ,:prec=>1e-1
);

Fω = [[ComplexF64[] for _ in 0:basis_field_order] for _ in Ω];
Tω = Array{ComplexF64}(undef,basis_field_order+1,length(Ω));
T0ω = Array{ComplexF64}(undef,basis_field_order+1,length(Ω));

progress = Progress(length(Ω));
@time Threads.@threads for i=1:length(Ω)
    ω = Ω[i];
    Fω[i] = sample_effective_t_matrix(ω, host_medium, sp_MC;kws_MC...);

    T, T0 = effective_T_matrices(ω,host_medium,[sp_EF];
                radius_big_cylinder=radius_big_cylinder, basis_order=basis_order, basis_field_order=basis_field_order)
    Tω[:,i] .= T[basis_field_order+1:2basis_field_order+1]
    T0ω[:,i] .= T0[basis_field_order+1:2basis_field_order+1]
    next!(progress)
end 

MCtemp_vec = Vector{MonteCarloResultTemp}(); #[];
MC_vec = Vector{MonteCarloResult}();
for i=1:length(Ω)
    push!(MCtemp_vec,MonteCarloResultTemp(basis_order,basis_field_order,Ω[i],sp_MC,radius_big_cylinder,Fω[i],Tω[:,i],T0ω[:,i]));
    push!(MC_vec,MonteCarloResult(MCtemp_vec[i]));
end

Fω,Tω,T0ω = 0,0,0; # clear some memory when done with this variable

save(MCtemp_vec,"Sound hard particles, loop on frequencies.")