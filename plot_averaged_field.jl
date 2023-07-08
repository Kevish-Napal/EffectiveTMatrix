using MultipleScattering
using EffectiveWaves
using Plots
# theme(:dracula) # requires pkg PlotThemes
using DataFrames, CSV
using ProgressMeter
using LaTeXStrings
using Dates

include("averaged_multipole_decomposition.jl")
include("common_params.jl")
include("EffectiveSphere.jl")
include("../../MyJuliaFunctions/MultipleScattering/functions.jl")



basis_order = 5;

ϕ = 0.05 # 
particle = Particle(Acoustic(2; ρ=Inf, c=Inf),Circle(1.0))
# particle = Particle(Acoustic(2; ρ=1e-2, c=1.0),Circle(1.0))
# particle = Particle(Acoustic(2; ρ=1.0, c=0.5),Circle(1.0))
Rtilde = radius_big_cylinder - particle.shape.radius
sp_MC, sp_EF = generate_species(radius_big_cylinder,particle,ϕ,separation_ratio)

c = 1.0
ω = .1;
λ = 2pi*c/ω;


## where to plot the field

M=N=2*radius_big_cylinder
res=100
bottomleft = [-M;-N]; topright = [M;N];
region = Box([bottomleft, topright]);
x_vec=points_in_shape(region;resolution=res);
println("λ=",λ,"\nparticle_radius=",particle.shape.radius,"\nmesh size=",x_vec[1][2][1]-x_vec[1][1][1])
########################## Source ##################################
mode = 3
source = mode_source(mode);
plot(source,ω;bounds=region,res=res)
plot!([Rtilde*(cos(t)+im*sin(t)) for t in 0:.1:2pi],linewidth=3,linecolor=:black,linestyle=:dash,legend=false)
plot_source = plot!(title="",guide="",ticks=false, colorbar=false)
# savefig("source_mode0"*string(mode)*".pdf") 


################# Average #############################
## loop on the different configurations
nb_of_configurations = 100;
A = complex(zeros(length(x_vec[1]),nb_of_configurations));

@time Threads.@threads for i=1:nb_of_configurations
    particles = renew_particle_configurations(sp_MC,radius_big_cylinder);
    sim = FrequencySimulation(particles,source);
    result_tot = run(sim,x_vec[1],[ω];only_scattered_waves=true,basis_order=basis_order)
    A[:,i] = mean.(result_tot.field[:,1]) # trick to get read of SVector
end 


nb_samples = 100 # nb_of_configurations
Tr_A  = mean(A[:,nb_of_configurations-nb_samples+1:nb_of_configurations],dims=2);
# Tr_A[norm.(x_vec[1]) .> Rtilde] .=0.0+0.0im
averaged_result_tot = FrequencySimulationResult(Tr_A,x_vec[1],[ω])


plot(averaged_result_tot,ω; field_apply=real,seriestype = :contour,c=:balance) 
plot!([Rtilde*(cos(t)+im*sin(t)) for t in 0:.1:2pi],linewidth=3,linecolor=:black,linestyle=:dash,legend=false)
plot_average = plot!(title="",guide="",axis=false,colorbar=false)




############ modes amplitudes ###############
input_mode = mode;
basis_field_order = 4;
nb_iterations = 100;

# particles = renew_particle_configurations(sp_MC,radius_big_cylinder);
F1 = mode_analysis(input_mode, ω, host_medium, sp_MC;
                radius_big_cylinder=radius_big_cylinder, 
                basis_order=basis_order, 
                basis_field_order=basis_field_order,
                nb_iterations=1);

F2 = mode_analysis(input_mode, ω, host_medium, sp_MC;
                radius_big_cylinder=radius_big_cylinder, 
                basis_order=basis_order, 
                basis_field_order=basis_field_order,
                nb_iterations=nb_iterations);

scatter(0:basis_field_order,abs.(mean.(F1)),label="1 realisation")
scatter!(0:basis_field_order,abs.(mean.(F2)),label=string(nb_iterations)*" realisations",
            yticks=false)

plot_amplitudes = scatter!(title="",
         xlabel="n",
         ylabel=L"$|\langle\mathfrak{F}_n\rangle|$",
         legend=:outertop)         
# savefig("Fn_"*string(nb_iterations)*"realisations_mode"*string(input_mode)*".pdf")      


## one realisation - draw convenient particles configuration
sim = FrequencySimulation(particles,source);
result_tot = run(sim,x_vec[1],[ω];only_scattered_waves=true,basis_order=basis_order)

plot(result_tot,ω; field_apply=real,seriestype = :contour,c=:balance) 
plot!([Rtilde*(cos(t)+im*sin(t)) for t in 0:.1:2pi],linewidth=3,linecolor=:black,linestyle=:dash,legend=false)
plot!(particles)
plot_realisation = plot!(title="",guide="",axis=false,colorbar=false)
# savefig("realisation_mode"*string(mode)*".pdf")


plot(plot_source, plot_realisation, plot_average, plot_amplitudes)
plot!(size=(500,500),title=["source" "1 realisation" "100 realisations" "excited modes"])
plot!(plot_title="mode N="*string(mode))
plot!(background_color=:lightgray)
savefig("modal_scattering_"*string(mode)*".pdf")


## mode field plots 
function transmitted_field(mode,ω,host_medium,sp_EF;radius_big_cylinder,basis_order,num_wavenumbers)
    
    k = ω/host_medium.c
    
    micro = Microstructure(host_medium,[sp_EF]);
    Rtilde = radius_big_cylinder - sp_EF.particle.shape.radius

    opts = Dict(
       :tol => 1e-4, 
       :num_wavenumbers => num_wavenumbers
       ,:basis_order => basis_order
   );
   
   kstar = wavenumbers(ω,micro;opts...)

    Snp =  [2.0im/(π*Rtilde)*
        ( k*besselh(mode-1,k*Rtilde)*besselj(mode,kp*Rtilde)-
            kp*besselh(mode,k*Rtilde)*besselj(mode-1,kp*Rtilde))^-1
            for kp in kstar]
    
    function u(x)
        r, θ = cartesian_to_radial_coordinates(x)
        return (norm(x)<Rtilde) ? dot(Snp,besselj.(mode,kstar*r)) * exp(im*mode*θ) : 0.0+0.0im
    end

    return kstar, u    
end

kstar, u = transmitted_field(mode,ω,host_medium,sp_EF;
    radius_big_cylinder=radius_big_cylinder,basis_order=5,
    num_wavenumbers=3)

u_vec = u.(x_vec[1]);

transfield = FrequencySimulationResult(u_vec,x_vec[1],[ω]);
plot(transfield,ω; field_apply=real,seriestype = :contour,c=:balance) 


transmitted_error = vec(mean(A[:,nb_of_configurations-nb_samples+1:nb_of_configurations],dims=2)) - u_vec;
transmitted_rel_error = transmitted_error./norm.(u_vec);

transmitted_error[norm.(x_vec[1]) .> Rtilde] .=0.0+0.0im
transmitted_rel_error[norm.(x_vec[1]) .> Rtilde] .=0.0+0.0im;

FS_transmitted_error = FrequencySimulationResult(transmitted_error,x_vec[1],[ω]);
plot(FS_transmitted_error,ω; field_apply=abs,seriestype = :contour,c=:balance) 

FS_transmitted_rel_error = FrequencySimulationResult(transmitted_rel_error,x_vec[1],[ω]);
plot(FS_transmitted_rel_error,ω; field_apply=abs,seriestype = :contour,c=:balance) 


cst = vec(mean(A[:,nb_of_configurations-nb_samples+1:nb_of_configurations],dims=2))./u_vec;
cst[norm.(x_vec[1]) .> 10] .= 0.05+0.175im;
FS_cst = FrequencySimulationResult(cst,x_vec[1],[ω]);
plot(FS_cst,ω; field_apply=imag,seriestype = :contour,c=:balance) 





function v(x)
    r, θ = cartesian_to_radial_coordinates(x)
    return besselj.(mode,kstar[3]*r) * exp(im*mode*θ)
end

kstar, u = transmitted_field(mode,ω,host_medium,sp_EF;
    radius_big_cylinder=radius_big_cylinder,basis_order=5,num_wavenumbers=4)

v_vec = v.(x_vec[1]);
v_vec[norm.(x_vec[1]) .> 1.5] .=0.0+0.0im;
field_v = FrequencySimulationResult(v_vec,x_vec[1],[ω])
plot(field_v,ω; field_apply=real,seriestype = :contour,c=:balance) 

function f(mode,k,x;field_apply=real)
    return field_apply(besselj(mode,k*x))
end

    
plot(x->f(0,10+1im,x),xlims=(-50,50))



