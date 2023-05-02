# Return matrix of computed coefficients a col corresponds to a configuration
using SpecialFunctions
using Statistics
using LinearAlgebra

 

function renew_particle_configurations(sp::Specie,radius_big_cylinder::Float64)
    config = random_particles(
        sp.particle.medium,
        sp.particle.shape;
        region_shape = Circle(1.05*radius_big_cylinder),
        volume_fraction = sp.volume_fraction
    );
    config = config[norm.(origin.(config)) .< radius_big_cylinder .- outer_radius.(config)]
end


"""
    sample_effective_t_matrix(ω::Number, host_medium::PhysicalMedium, sp::Specie, select_modes::Vector{Bool};kws...)

Monte Carlo computation of the effective cylinder T_matrix. The coefficients are computed for n 
in [-basis_field_order::basis_field_order]
"""

# function sample_effective_t_matrix(ω::Number, host_medium::PhysicalMedium, sp::Specie;
#     radius_big_cylinder=10.0::Float64, basis_order=10::Int, basis_field_order=0::Int,
#     nb_iterations_max=50000::Int,nb_iterations_step=200::Int,prec=1e-1::Float64) 

#     k = ω/host_medium.c
#     CB = regular_basis_function(k, host_medium)
#     # Precompute T-matrices for these particles
#     t_matrix = get_t_matrices(host_medium, [sp.particle], ω, basis_order)[1]

#     # We need this big vector F to store realisation of each modes 
#     # which convergent rate my differ
#     F = [ComplexF64[] for _ in 1:2*basis_field_order+1]

#     total_iterations = Int.(zeros(2*basis_field_order+1))
    
#     μ = complex(zeros(2*basis_field_order+1))
#     σ_r = zeros(2*basis_field_order+1)
#     σ_i = zeros(2*basis_field_order+1)
#     fiability = 2.0 # 2 => 95% chances to lie in the interval of confidence 

#     select_modes = trues(2*basis_field_order+1)

#     continue_crit() = any(select_modes) && maximum(total_iterations) < nb_iterations_max
    
#     # confidence interval level 95% is [m-1.96σ/√n ; m+1.96σ/√n]
#     # we have [1.96σ/√nb_iterations < prec] => μ = ;empirical_mean ± precision] 
#     mode_continue_crit(m,s_r,s_i,n,prec) = (1.96*s_r/sqrt(n) > prec || 1.96*s_i/sqrt(n) > prec) && abs(m) > 1e-8


#     while continue_crit()

#         println("iteration:\n",total_iterations)
#         println("modes still running:\n", select_modes)

#         for _ in 1:nb_iterations_step

#             particles = renew_particle_configurations(sp,radius_big_cylinder)
#             n_particles = length(particles)

#             # Computation of all bessel functions - this can be optimized following selected_modes 
#             M_bessel = reduce(hcat, [CB(basis_order+basis_field_order,origin(p)) for p in particles])

#             # Compute scattering matrix for all particles
#             S = scattering_matrix(host_medium, particles, [t_matrix for p in particles], ω, basis_order)

#             V = Array{ComplexF64,2}(undef,2*basis_order+1,n_particles)
#             a = Array{ComplexF64,2}(undef,2*basis_order+1,n_particles)

      
#             for N = 1:2*basis_field_order+1 # optimal when this loop comes after renewing particles
#                 if  select_modes[N]

#                     # V_{N-n} for n = -basis_order:basis_order
#                     V .= M_bessel[N+2*basis_order:-1:N,:]
                
                    
#                     # reshape and multiply by t-matrix to get the scattering coefficients
#                     a .= reshape((S + I) \ reduce(vcat,V),2*basis_order+1,n_particles)


#                     for i in axes(a,2)
#                         a[:,i] = t_matrix * a[:,i]
#                     end
                    
#                     # this uses a lot of memory, should be optimized
#                     F_step= sum(conj(V).*a)
#                     push!(F[N],F_step)
                    
#                 end
#             end                                                      # end mode loop
#         end                                                          # end iteration step
       
#         # recompute mean values and stdm for extended vector Fn (more realisations)
#         for N = 1:2*basis_field_order+1 
#             if select_modes[N]

#                 total_iterations[N] += nb_iterations_step
#                 # println(total_iterations)

#                 m = mean(F[N])
#                 μ[N] = m
#                 M_σ = F[N].-m                     
        
#                 s_r = sqrt.(mean((real.(M_σ).^2)))
#                 σ_r[N] = s_r 
                
#                 s_i = sqrt.(mean((imag.(M_σ).^2)))
#                 σ_i[N] = s_i  


#                 select_modes[N] = mode_continue_crit(m,s_r,s_i,total_iterations[N],prec)
#             end
#         end
#     end                                                              # end while
#     return  μ # MonteCarloResult(ω,sp,radius_big_cylinder,μ,σ_r+im*σ_i,total_iterations)
# end


function sample_effective_t_matrix(ω::Number, host_medium::PhysicalMedium, sp::Specie;
    radius_big_cylinder=10.0::Float64, basis_order=10::Int, basis_field_order=0::Int,
    nb_iterations_max=50000::Int,nb_iterations_step=200::Int,prec=1e-1::Float64) 

    k = ω/host_medium.c
    # Precompute T-matrices for these particles
    t_matrix = get_t_matrices(host_medium, [sp.particle], ω, basis_order)[1]

    # We need this big vector F to store realisation of each modes 
    # which convergent rate my differ
    F = [ComplexF64[] for _ in 1:basis_field_order+1]

    total_iterations = Int.(zeros(basis_field_order+1))
    
    μ = complex(zeros(basis_field_order+1))
    σ_r = zeros(basis_field_order+1)
    σ_i = zeros(basis_field_order+1)
    

    initial_basis_field_order = basis_field_order
    select_modes = trues(basis_field_order+1)

    continue_crit() = any(select_modes) && maximum(total_iterations) < nb_iterations_max
    
    # confidence interval level 95% is [m-1.96σ/√n ; m+1.96σ/√n]
    # we have [1.96σ/√nb_iterations < prec] => μ = ;empirical_mean ± precision] 
    mode_continue_crit(m,s_r,s_i,n,prec) = (1.96*s_r/sqrt(n) > prec || 1.96*s_i/sqrt(n) > prec) && abs(m) > 1e-8


    while continue_crit()

        println("iteration:\n",total_iterations)
        println("modes still running:\n", select_modes)

        for _ in 1:nb_iterations_step

            particles = renew_particle_configurations(sp,radius_big_cylinder)
            rθ = [cartesian_to_radial_coordinates(origin(p)) for p in particles]
            n_particles = length(particles)
            J = [
                besselj(n,k*rθ[i][1]) 
            for n = 0:(basis_order+basis_field_order),i=1:n_particles]
            M_bessel = [
                (-1)^n*J[abs(n)+1,i]*exp(im*n*rθ[i][2]) 
            for n=-basis_order-basis_field_order:basis_order+basis_field_order,i=1:n_particles]

            # Computation of all bessel functions - this can be optimized following selected_modes 
            

            # Compute scattering matrix for all particles
            S = scattering_matrix(host_medium, particles, [t_matrix for p in particles], ω, basis_order)

            V = Array{ComplexF64,2}(undef,2*basis_order+1,n_particles)
            a = Array{ComplexF64,2}(undef,2*basis_order+1,n_particles)

      
            for N = 1:(basis_field_order + 1) # optimal when this loop comes after renewing particles
                if  select_modes[N]

                    # V_{N-n} for n = -basis_order:basis_order
                    inds = (N + 2*basis_order + basis_field_order):-1:(N + basis_field_order)
                    V .= M_bessel[inds,:]  
                
                    
                    # reshape and multiply by t-matrix to get the scattering coefficients
                    a .= reshape((S + I) \ reduce(vcat,V),2*basis_order+1,n_particles)


                    for i in axes(a,2)
                        a[:,i] = t_matrix * a[:,i]
                    end
                    
                    # this uses a lot of memory, should be optimized
                    F_step= sum(conj(V).*a)
                    push!(F[N],F_step)
                    
                end
            end                                                      # end mode loop
        end                                                          # end iteration step
       
        # recompute mean values and stdm for extended vector Fn (more realisations)
        for N = 1:basis_field_order+1
            if select_modes[N]

                total_iterations[N] += nb_iterations_step
                # println(total_iterations)

                m = mean(F[N])
                μ[N] = m
                M_σ = F[N].-m                     
        
                s_r = sqrt.(mean((real.(M_σ).^2)))
                σ_r[N] = s_r 
                
                s_i = sqrt.(mean((imag.(M_σ).^2)))
                σ_i[N] = s_i  


                select_modes[N] = mode_continue_crit(m,s_r,s_i,total_iterations[N],prec)
            end
        end
        if select_modes[1]
            basis_field_order = maximum(collect(0:initial_basis_field_order)[select_modes])
        end
    end       
                                                           # end while
    if initial_basis_field_order >= 1
        μ = vcat(reverse(μ[2:end]),μ)
    end

    return F   # MonteCarloResult(ω,sp,radius_big_cylinder,μ,σ_r+im*σ_i,total_iterations)
end



mutable struct MonteCarloResult
    ω::Float64
    sp::Specie
    R::Float64

    μ::Vector{ComplexF64}
    σ::Vector{ComplexF64}
    nb_iterations::Vector{Int64}

    μeff::Vector{ComplexF64}
    μeff0::Vector{ComplexF64}

    function MonteCarloResult(
        ω::Float64,
        sp::Specie,
        R::Float64,

        μ::Vector{ComplexF64},
        σ::Vector{ComplexF64},
        nb_iterations::Vector{Int64},

        μeff::Vector{ComplexF64},
        μeff0::Vector{ComplexF64})

        if length(μ) != length(σ) || length(μ) != length(nb_iterations) || length(σ) != length(nb_iterations)
            error("μ, σ and nb_iterations have to be of same length")
        end

        new(ω,sp,R,μ,σ,nb_iterations,μeff,μeff0)
    end
end

function MonteCarloResult(basis_field_order::Int,sp::Specie) 
    N = 2*basis_field_order+1
    MonteCarloResult(1.0,sp,1.0,Array{ComplexF64}(undef,N),Array{ComplexF64}(undef,N),Array{Int}(undef,N),
    Array{ComplexF64}(undef,N),Array{ComplexF64}(undef,N))
end

function update!_Monte_Carlo_Result(F::MonteCarloResult,Ftemp::MonteCarloResult)

    if length(F.μ)!=length(Ftemp.μ)
        error("size mismatch: try renewing kws_MC... if basis_field_order changed.")
    end

    F.ω = Ftemp.ω
    F.sp = Ftemp.sp
    F.R = Ftemp.R
    F.μ .= Ftemp.μ
    F.σ .= Ftemp.σ
    F.nb_iterations .= Ftemp.nb_iterations
end

function update!_Effective_Result(F::MonteCarloResult, μeff::Vector{ComplexF64}, μeff0::Vector{ComplexF64})

    if length(F.μ)!=length(μeff) || length(F.μ)!=length(μeff0)
        error("size mismatch: try renewing kws_MC... if basis_field_order changed.")
    end

    F.μeff .= μeff
    F.μeff0 .= μeff0
end

function update!_Effective_Result(F::MonteCarloResult,basis_order,host_medium)

    kstar, wavemode = effective_sphere_wavenumber(F.ω,[sp_MC_to_EF(F)],host_medium;
    radius_big_cylinder=F.R,basis_order=basis_order);
    N,D = t_matrix_num_denom(kstar,wavemode;basis_field_order=(length(F.μ)-1)/2);
    T =  - vec(sum(N,dims=1)./sum(D,dims=1))
    T0 = - N[1+basis_order,:]./D[1+basis_order,:]

    F.μeff .= T
    F.μeff0 .= T0
end



function sp_MC_to_EF(F_MC::MonteCarloResult)

    sp_MC = F_MC.sp
    particle_radius = outer_radius(sp_MC.particle)
    radius_big_cylinder = F_MC.R
    ϕ = F_MC.mean_nb_particles*particle_radius^2/(radius_big_cylinder-particle_radius)^2
    
    return Specie(Acoustic(2; ρ=sp_MC.particle.medium.ρ, c=sp_MC.particle.medium.c),Circle(particle_radius);
    volume_fraction = ϕ,separation_ratio=F_MC.sp.separation_ratio)
end

function sp_MC_to_EF(sp_MC::Specie,radius_big_cylinder::Float64)

    mean_nb_particles = mean([length(renew_particle_configurations(sp_MC,radius_big_cylinder)) for i = 1:500])
    particle_radius = outer_radius(sp_MC.particle)
    ϕ = mean_nb_particles*particle_radius^2/(radius_big_cylinder-particle_radius)^2
    
    return Specie(Acoustic(2; ρ=sp_MC.particle.medium.ρ, c=sp_MC.particle.medium.c),Circle(particle_radius);
    volume_fraction = ϕ,separation_ratio=sp_MC.separation_ratio)
end

function generate_species(radius_big_cylinder::Float64,particle::Particle,ϕ::Float64,separation_ratio::Float64)

    sp_MC = Specie(particle; volume_fraction = ϕ,separation_ratio=separation_ratio) 
    sp_EF = sp_MC_to_EF(sp_MC,radius_big_cylinder)

    return sp_MC, sp_EF
end


function MC_to_df(F0_MC::Vector{MonteCarloResult})
    N_df = length(F0_MC)
    basis_field_order = Int((length(F0_MC[1].μ)-1)/2)
    df = DataFrame()
    df.ω = [F0_MC[i].ω for i=1:N_df]
    df.R = [F0_MC[i].R for i=1:N_df]

    df.particle_radius = [outer_radius(F0_MC[i].sp.particle) for i=1:N_df]
    df.c_particle = [F0_MC[i].sp.particle.medium.c for i=1:N_df]
    df.ρ_particle = [F0_MC[i].sp.particle.medium.ρ for i=1:N_df]
    df.separation_ratio = [F0_MC[i].sp.separation_ratio for i=1:N_df]
    df.volume_fraction = [F0_MC[i].sp.volume_fraction for i=1:N_df]

    df.μ = [F0_MC[i].μ[basis_field_order+1] for i=1:N_df]
    df.σ = [F0_MC[i].σ[basis_field_order+1] for i=1:N_df]
    df.nb_iterations = [F0_MC[i].nb_iterations for i=1:N_df]

    df.μeff = [F0_MC[i].μeff[basis_field_order+1] for i=1:N_df]
    df.μeff0 = [F0_MC[i].μeff0[basis_field_order+1] for i=1:N_df]
    return df
end

function MC_to_df(F0_MC::MonteCarloResult)
    return df_output([F0_MC])
end

function df_to_MC(df)

    Nb_MC = length(df.ω)
    F = Array{MonteCarloResult,1}(undef,Nb_MC)
    for i = 1:Nb_MC
        particle = Particle(Acoustic(2; ρ=df.ρ_particle[i], c=fd.c_particle),Circle(df.particle_radius[i]))
        sp_MC = Specie(particle; volume_fraction = df.volume_fraction[i],separation_ratio=df.separation_ratio[i]) 
        update!_Monte_Carlo_Result(F[i], 
                                            MonteCarloResult(
                                                                df.ω[i],
                                                                sp_MC,
                                                                df.R[i],

                                                                [df.μ[i]],
                                                                [df.sem[i]],
                                                                [df.nb_iterations[i]],

                                                                [df.μeff[i]],
                                                                [df.μeff0[i]]
                                                            )
        )
    end
end

function Base.show(io::IO,MC::MonteCarloResult)
    basis_field_order = Int((length(MC.μ)-1)/2)
    stdm_r = real.(MC.σ)./ sqrt.(MC.nb_iterations)
    stdm_i = imag.(MC.σ)./ sqrt.(MC.nb_iterations)
    N = length(MC.μ)

    print(io,"----------------------------------------------------------------------------------------------------")
    print(io, 
        "
        ω = $(MC.ω)
        container radius: $(MC.R)
        Particle type: $(MC.sp.particle) 
        Values:
     ")

     for n = 1:N
        print(io, 
        "
        F_$(n-basis_field_order-1) =  $(MC.μ[n]) ± $(stdm_r[n]+im*stdm_i[n]) ($(MC.nb_iterations[n]) iterations) 
        ")
     end

     print(io,"----------------------------------------------------------------------------------------------------")
end


##### Effective wavenumber

struct EffectiveSphere{T,Dim,M<:Microstructure} <: AbstractParticle{Dim} where T<:AbstractFloat
    material::Material{Sphere{T,Dim},M}
end


function t_matrix(
    effective_sphere::EffectiveSphere{T,2},outer_medium::Acoustic{T,2}, ω::T;
    kws...)::Diagonal{Complex{T}} where T <: AbstractFloat
    return t_matrix(effective_sphere.material,outer_medium, ω; kws...)
end


function effective_sphere_wavenumber(ω::Number,sps::Species,outer_medium::Acoustic{T,2};
    radius_big_cylinder=10.0::Float64,basis_order=10::Int) where {T,P}

    micro = Microstructure(outer_medium,sps);
    material = Material(Circle(radius_big_cylinder),micro);

    opts = Dict(
       :tol => 1e-4, 
       :num_wavenumbers => 1
       ,:basis_order => basis_order
   );
   
   kstar = wavenumbers(ω,micro;opts...)[1]

   kws = Dict(
       :basis_order => basis_order
       ,:tol=>1e-2
   );
   
   rsource = regular_spherical_source(outer_medium, [1.0+0.0im];
   position = zeros(dimension), symmetry = RadialSymmetry{dimension}()
   );

   wavemode = WaveMode(ω, kstar, rsource, material;kws...);

   return kstar, wavemode
end 

function t_matrix_num_denom(kstar,wavemode;basis_field_order)
    
    k=wavemode.ω/wavemode.medium.c
    R = wavemode.material.shape.radius
    species = wavemode.material.microstructure.species
    Rtildas = R .- outer_radius.(species)

    nbo, n_λ, _ = size(wavemode.eigenvectors)
    basis_order = Int((nbo-1)/2)
    L = basis_order+basis_field_order

    F = wavemode.eigenvectors

    n_densities = [number_density(sp) for sp in species]

    J = [besselj(n,k*Rtildas[λ])*n_densities[λ] 
        for n = -L-1:L,
            λ in 1:n_λ]

    Jstar = [besselj(n,kstar*Rtildas[λ])*n_densities[λ] 
        for n = -L-1:L,
            λ in 1:n_λ]

    H = [besselh(n,k*Rtildas[λ])*n_densities[λ] 
        for n = -L-1:L,
            λ in 1:n_λ]

    pre_num = k*J[1:1+2*L,:].*Jstar[2:2*(1+L),:] -
              kstar*J[2:2*(1+L),:].*Jstar[1:1+2*L,:]
    pre_denom = k*H[1:1+2*L,:].*Jstar[2:2*(1+L),:] -
                kstar*H[2:2*(1+L),:].*Jstar[1:1+2*L,:]

    Matrix_Num = complex(zeros(2*basis_order+1,2*basis_field_order+1))
    Matrix_Denom = complex(zeros(2*basis_order+1,2*basis_field_order+1))

    for n = 1:1+2*basis_field_order
        
        Matrix_Num[:,n] = vec(sum(pre_num[n+2*basis_order:-1:n,:].*F,dims=2))

        Matrix_Denom[:,n] = vec(sum(pre_denom[n+2*basis_order:-1:n,:].*F,dims=2))
    end

    return Matrix_Num, Matrix_Denom
end

# main function to compute effective T-matrix
function t_matrix(ω::T,effective_cylinder::Material,outer_medium::Acoustic{T,2};
    basis_order=10::Integer,basis_field_order=10::Integer,include_terms=basis_order::Int) where T <: AbstractFloat

    radius_big_cylinder = effective_cylinder.shape.radius
    micro = effective_cylinder.microstructure
    species = micro.species
#    rs = outer_radius.(species)

#    R = effective_sphere.shape.radius 
#    k=ω/outer_medium.c

    # Check for material properties that don't make sense or haven't been implemented
#    for sp in species
#        check_material(sp.particle, outer_medium)
#    end

    kstar, wavemode =  effective_sphere_wavenumber(ω,species,outer_medium ;
    basis_order= basis_order,radius_big_cylinder=radius_big_cylinder)


#    numer(m,R) = k * diffbesselj(m, R*k) * besselj(m, R*kstar) - kstar * besselj(m, R*k)*diffbesselj(m, R*kstar)
#    denom(m,R) = k * diffhankelh1(m, R*k) * besselj(m, R*kstar) - kstar * hankelh1(m, R*k)*diffbesselj(m, R*kstar)
   
#    Nn = [sum(
#        numer(i[1]-1-input_basis_order-n,R-rs[i[2]]) *
#        wavemode.eigenvectors[i] *
#        number_density(species[i[2]])
#    for i in CartesianIndices(wavemode.eigenvectors))
#        for n = -output_basis_order:output_basis_order]

#    Dn = [sum(
#        denom(i[1]-1-input_basis_order-n,R-rs[i[2]]) *
#        wavemode.eigenvectors[i] *
#        number_density(species[i[2]])
#    for i in CartesianIndices(wavemode.eigenvectors))
#        for n = -output_basis_order:output_basis_order]

   N, D = t_matrix_num_denom(kstar,wavemode;basis_field_order = basis_field_order)

   return - vec(sum(N[basis_order+1-include_terms:basis_order+1+include_terms,:],dims=1) ./ sum(D[basis_order+1-include_terms:basis_order+1+include_terms,:],dims=1))
end



## for Plots

function twiny(sp::Plots.Subplot)
    sp[:top_margin] = max(sp[:top_margin], 30Plots.px)
    plot!(sp.plt, inset = (sp[:subplot_index], bbox(0,0,1,1)))
    twinsp = sp.plt.subplots[end]
    twinsp[:xaxis][:mirror] = true
    twinsp[:background_color_inside] = RGBA{Float64}(0,0,0,0)
    Plots.link_axes!(sp[:yaxis], twinsp[:yaxis])
    twinsp
end
twiny(plt::Plots.Plot = current()) = twiny(plt[1])