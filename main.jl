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


### generate data

# Ω = collect(0.01:0.025:.5);
basis_order = 5;
basis_field_order = 4;

ϕ = 0.05 # 
particle = Particle(Acoustic(2; ρ=1e-2, c=1.0),Circle(1.0))
# particle = Particle(Acoustic(2; ρ=Inf, c=Inf),Circle(1.0))
sp_MC, sp_EF = generate_species(radius_big_cylinder,particle,ϕ,separation_ratio)

kws_MC = Dict(
    :radius_big_cylinder=>radius_big_cylinder
    ,:basis_order=> basis_order
    ,:basis_field_order=> basis_field_order
    ,:nb_iterations_max=> 5000
    ,:nb_iterations_step=> 100
    ,:prec=>5e-2
);

Fω = [[ComplexF64[] for _ in 0:basis_field_order] for _ in Ω];
Tω = Array{ComplexF64}(undef,basis_field_order+1,length(Ω));
T0ω = Array{ComplexF64}(undef,basis_field_order+1,length(Ω));

# progress = Progress(length(Ω));
# @time Threads.@threads for i=1:length(Ω)
@time for i=1:length(Ω)
    println(i)
    ω = Ω[i]
    Fω[i] = sample_effective_t_matrix(ω, host_medium, sp_MC;kws_MC...);

    T, T0 = effective_T_matrices(ω,host_medium,[sp_EF];
                radius_big_cylinder=radius_big_cylinder, basis_order=basis_order, basis_field_order=basis_field_order)
    Tω[:,i] .= T[basis_field_order+1:2basis_field_order+1]
    T0ω[:,i] .= T0[basis_field_order+1:2basis_field_order+1]
    # next!(progress)
end 

MCtemp_vec = Vector{MonteCarloResultTemp}(); #[];
MC_vec = Vector{MonteCarloResult}();
for i=1:length(Ω)
    push!(MCtemp_vec,MonteCarloResultTemp(basis_order,basis_field_order,Ω[i],sp_MC,radius_big_cylinder,Fω[i],Tω[:,i],T0ω[:,i]));
    push!(MC_vec,MonteCarloResult(MCtemp_vec[i]));
end

Fω,Tω,T0ω = 0,0,0; # clear some memory when done with this variable

save(MCtemp_vec,"Sound soft particles, loop on a high frequencies.")



### load parameters of saved data

ω, radius_big_cylinder, basis_order, basis_field_order, sp_MC, sp_EF = load_parameters(4);
Ω = collect(0.515:0.015:1.5)

kws_MC = Dict(
    :radius_big_cylinder=>radius_big_cylinder
    ,:basis_order=> basis_order
    ,:basis_field_order=> basis_field_order
    ,:nb_iterations_max=> 5000
    ,:nb_iterations_step=> 400
    ,:prec=>5e-2
);

Fω = [[ComplexF64[] for _ in 0:basis_field_order] for _ in Ω];
Tω = Array{ComplexF64}(undef,basis_field_order+1,length(Ω));
T0ω = Array{ComplexF64}(undef,basis_field_order+1,length(Ω));

for i=1:length(Ω)
    println(i)
    ω = Ω[i]
    Fω[i] = sample_effective_t_matrix(ω, host_medium, sp_MC;kws_MC...);

    T, T0 = effective_T_matrices(ω,host_medium,[sp_EF];
                radius_big_cylinder=radius_big_cylinder, basis_order=basis_order, basis_field_order=basis_field_order)
    Tω[:,i] .= T[basis_field_order+1:2basis_field_order+1]
    T0ω[:,i] .= T0[basis_field_order+1:2basis_field_order+1]
end 

MCtemp_vec = Vector{MonteCarloResultTemp}(); #[];
MC_vec = Vector{MonteCarloResult}();
for i=1:length(Ω)
    push!(MCtemp_vec,MonteCarloResultTemp(basis_order,basis_field_order,Ω[i],sp_MC,radius_big_cylinder,Fω[i],Tω[:,i],T0ω[:,i]));
    push!(MC_vec,MonteCarloResult(MCtemp_vec[i]));
end

Fω,Tω,T0ω = 0,0,0; # clear some memory when done with this variable

save(MCtemp_vec,"Sound hard particles, loop on a high frequencies.")

plot(MC_vec)


for mode = 0:4
    plot(MC_vec_tot;mode=mode)
    savefig("Neuman"*string(mode)*".pdf")
end


##################### Load data ########################## 
MC_vec1 = MC_read(7);
MC_vec2 = MC_read(8);
MC_vec = [MC_vec1;MC_vec2];
plot(MC_vec)

save(MC_vec,"Sound soft particles, loop on broad range of frequencies.")


################## plot Dirichlet and Neumann ########################## 
MC_vec_dir = MC_read(9);
MC_vec_neu = MC_read(6);
Ω = [MC.ω for MC in MC_vec_neu]
plot(MC_vec_neu;MC_filter=1,
MC_color=:black,MA_color=:green,exact_color=:coral)

# There is too many simulations


plot([MC_vec_dir,MC_vec_neu])
plot!(background_color=:lightgray)
savefig("D&N.pdf")