using MultipleScattering
using EffectiveWaves
using Plots
using DataFrames, CSV
using ProgressMeter
using LaTeXStrings

include("averaged_multipole_decomposition.jl")
include("common_params.jl")

################################ test 1: is the optimized code working? ######################################
# We compute <Fnn> (average nth mode of the scattered field when the incident field is such that gN = δn,N)
# This quantity can be computed in two different ways:
# 1 - use the optimized code for the computation of the effective T-matrix nth mode 
# 2 - still with gN = δn,N ,  the average scattered field satisfies
# ...                  <us>(x,0) = <Fnn> Hn(k x), for x>2R.
# ...  <Fnn> can then be extracted from the average scattered field. 
# Note: in Monte Carlo simulations, the formula above is corrupted by other modes of the scattered field (which 
# vanishes after convergence of the average). A possible refinement is to compute the nth Fourier mode of the computed <us>.


    basis_order = 3
    basis_field_order = 1

    nb_iterations = 100
    kws_MC = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
        ,:nb_iterations_max=> nb_iterations
        ,:nb_iterations_step=> nb_iterations
        ,:prec=>1e-2
    )

    ω = 0.5
    k = ω2k(ω)
    Fopt = sample_effective_t_matrix(ω, host_medium, sp_MC;kws_MC...)

# Here is the naive method using <us>
    Fnaive = [ComplexF64[] for _ in 1:basis_field_order+1]
    progress = Progress(basis_field_order+1)
    x = 1.5*radius_big_cylinder
   
    @time Threads.@threads for mode=0:basis_field_order
       source = mode_source(mode)
       for _ = 1:nb_iterations
        particles = renew_particle_configurations(sp_MC,radius_big_cylinder)
        sim = FrequencySimulation(particles,source);
        result = run(sim,[[x,0.0]],[ω];only_scattered_waves=true,basis_order=basis_order)
        Fnn = result.field[1][1]/besselh(mode,k*x)
        push!(Fnaive[mode+1],Fnn)
       end
       next!(progress)
    end 

    Fnaive = naive_sample_effective_t_matrix(ω, host_medium, sp_MC;
        radius_big_cylinder=radius_big_cylinder, 
        basis_order=basis_order, 
        basis_field_order=basis_field_order,
        nb_iterations=nb_iterations)
    

    mode = 1
    hr = histogram(real.(Fnaive[mode+1]),label="naive")
    hr = histogram!(real.(Fopt[mode+1]),label="optimal")

    hi = histogram(imag.(Fnaive[mode+1]),label="naive")
    hi = histogram!(imag.(Fopt[mode+1]),label="optimal")

    plot(hr,hi)
    err = abs(mean(Fopt[mode+1]) - mean(Fnaive[mode+1]))
    plot!(title=["mode="*string(mode),"image"])

    mean(Fopt[2])
    mean(Fnaive[2])
### END OF TEST 1






    result = run(sim,region,[ω];only_scattered_waves=true,basis_order=basis_order)

# radial source, compute F0 for different frequencies

    basis_order = 10
    basis_field_order = 0

    nb_iterations = 300
    kws_MC = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
        ,:nb_iterations_max=> nb_iterations
        ,:nb_iterations_step=> nb_iterations
        ,:prec=>1e-2
    )

    kws_EF = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
    )

    Ω = collect(0.1:0.1:1.0) # frqs


    μ = complex(zeros(nb_iterations,length(Ω)))
    progress = Progress(length(Ω))
    @time Threads.@threads for i=1:length(Ω)
        ω = Ω[i]
        μ[:,i] .= sample_effective_t_matrix(ω, host_medium, sp_MC;kws_MC...)[1]
        next!(progress)
     end 
    cummeans = cumsum(μ,dims=1) ./ collect(1:nb_iterations)
    YR_min = minimum(real.(cummeans))
    YR_max = maximum(real.(cummeans))
    YI_min = minimum(imag.(cummeans))
    YI_max = maximum(imag.(cummeans))

    ar = .1*(YR_max - YR_min); ai = .1*(YI_max - YI_min)
    YR_min = YR_min - ar
    YR_max = YR_max + ar
    YI_min = YI_min - ai
    YI_max = YI_max + ai

    T = complex(zeros(length(Ω)));
    T0 = complex(zeros(length(Ω)));

    progress = Progress(length(Ω))
    @time Threads.@threads for i=1:length(Ω)
        ω = Ω[i]
        kstar, wavemode = effective_sphere_wavenumber(ω,[sp_EF],host_medium;
        radius_big_cylinder=20.0,basis_order=basis_order);
        N,D = t_matrix_num_denom(kstar,wavemode;basis_field_order=basis_field_order);
        T[i] =  (- vec(sum(N,dims=1)./sum(D,dims=1)))[1]
        T0[i] = (- N[1+basis_order,:]./D[1+basis_order,:])[1]
    end

    ka = outer_radius(sp_MC)/real(host_medium.c) * Ω
    kR = radius_big_cylinder/real(host_medium.c) * Ω
    
    # plot(ka,real.(F),grid=false,ribbon=σ_r,fillalpha=.5)
    # plot(ka,imag.(F),grid=false,ribbon=σ_i,fillalpha=.5)
    # Real
    gr()
    anim = @animate for i=1:nb_iterations
        scatter(ka,real.(T),label=L"$\mathfrak{\Re\! e}(\mathcal{T}_0^{EF})$",xlabel="ka",color=:red,legend=:bottom)

        scatter!(twiny(),kR,real.(cummeans[i,:]),grid=false,label=L"$\mathfrak{\Re\! e}(\mathcal{T}_0^{MC})$",
            color=:green,legend=:bottomleft,yticks=:none,xlabel="\n kR")

        p_r = scatter!(ka,real.(T0),label=L"$\mathfrak{\Re\! e}(\mathcal{T}_0^{EF0})$",legend=:bottomright,
            title="iteration "*string(i)*"/"*string(nb_iterations),
            ylims=(YR_min,YR_max),color=:blue)
        
        ###
        scatter(ka,imag.(T),label=L"$\mathfrak{\Im\! m}(\mathcal{T}_0^{EF})$",xlabel="ka",color=:red,legend=:top)

        scatter!(twiny(),kR,imag.(cummeans[i,:]),grid=false,label=L"$\mathfrak{\Im\! m}(\mathcal{T}_0^{MC})$",
            color=:green,legend=:topleft,yticks=:none,xlabel="\n kR")

        p_i = scatter!(ka,imag.(T0),label=L"$\mathfrak{\Im\! m}(\mathcal{T}_0^{EF0})$",legend=:topright,
            title="iteration "*string(i)*"/"*string(nb_iterations),
            ylims=(YI_min,YI_max),color=:blue)
        ###
        
        scatter(ka,abs.(cummeans[i,:]-T),label=L"$|\mathcal{T}_0^{MC}-\mathcal{T}_0^{EF}|$",xlabel="ka",color=:red,legend=:topleft)

        scatter!(twiny(),kR,abs.(cummeans[i,:]-T0),grid=false,label=L"$|\mathcal{T}_0^{MC}-\mathcal{T}_0^{EF0}|$",
            color=:green,legend=:topright,yticks=:none,xlabel="\n kR")

        p_error = scatter!(title="\n absolute error",ylims=(0.0,0.3))

        # p_i = scatter!(ka,imag.(T0),label=L"$\mathfrak{\Im\! m}(\mathcal{T}_0^{EF0})$",legend=:topright,
        #     title="iteration "*string(i)*"/"*string(nb_iterations),
        #     ylims=(YI_min,YI_max),color=:blue)
            
            # scatter(twiny(),kR,abs.(T0-cummeans[i,:]),
        # label=L"$|\mathcal{T}_0^{MC}-\mathcal{T}_0^{EF0}|$",legend=:topright,
        # grid=false,color=:blue,yticks=:none,xlabel="\n kR")

        # p_error = scatter!(title="\n absolute error",ylims=(0.0,0.3))

         plot(p_r,p_i,p_error,size=(700,1000),markersize=3.5)
    end
    gif(anim, "convergence.gif", fps = 5)



    
    anim_i = @animate for i=1:nb_iterations
        scatter(ka,imag.(T),label=false,xlabel="ka",color=:red,legend=:bottomleft,title=L"$\mathfrak{\Re\! e}(\mathfrak{F}_0)$"*"\n\n")
        scatter!(twiny(),kR,imag.(cummeans[i,:]),grid=false,label="MC",color=:blue,legend=:bottom,yticks=:none,xlabel="\n kR")
        scatter!(ka,imag.(T0),label=false,title=string(i),ylims=(YI_min,YI_max))
    end
    gif(anim_i, fps = 5)


    anim_error = @animate for i=1:nb_iterations
        scatter(ka,abs.(T-cummeans[i,:]),label="EF",xlabel="ka",color=:red,legend=:bottomleft,title=L"$\mathfrak{\Re\! e}(\mathfrak{F}_0)$"*"\n\n")
        scatter!(twiny(),kR,abs.(T0-cummeans[i,:]),grid=false,label="MC",color=:blue,legend=:bottom,yticks=:none,xlabel="\n kR")
        scatter!(title=string(i),ylims=(0.0,0.3))
    end
    gif(anim_error, fps = 5)

    ###
    
    
    F0_MC = [MonteCarloResult(basis_field_order,sp_EF) for _ in Ω]
    progress = Progress(length(Ω))
   
    @time Threads.@threads for i=1:length(Ω)
       ω = Ω[i]
       update!_Monte_Carlo_Result(F0_MC[i],sample_effective_t_matrix(ω, host_medium, sp_MC;kws_MC...)) 
       next!(progress)
    end 

    df = df_output(F0_MC)
    CSV.write("test1.csv",df)

    F0_EF = complex(zeros(length(Ω)))
    progress = Progress(length(Ω))
    @time Threads.@threads for i=1:length(Ω)
        ω = Ω[i]
        F0_EF[i] = t_matrix(ω,effective_cylinder,host_medium;
            basis_order=basis_order,basis_field_order=basis_field_order)[1]
        next!(progress)
     end 
     
    F = [F0_MC[i].μ[basis_field_order+1] for i=1:length(Ω)]
    σ = [F0_MC[i].σ[basis_field_order+1] for i=1:length(Ω)]
    σ_r = real.(σ)
    σ_i = imag.(σ)
    

    ka = outer_radius(sp_MC)/real(host_medium.c) * Ω
    kR = radius_big_cylinder/real(host_medium.c) * Ω
    
    plot(ka,real.(F),grid=false,ribbon=σ_r,fillalpha=.5)
    plot(ka,imag.(F),grid=false,ribbon=σ_i,fillalpha=.5)
    # Real
    gr()
    scatter(ka,real.(F0_EF),label="EF",xlabel="ka",color=:red,legend=:bottomleft,title=L"$\mathfrak{\Re\! e}(\mathfrak{F}_0)$"*"\n\n")
    pR = plot!(twiny(),kR,real.(F),grid=false,ribbon=σ_r,fillalpha=.3,label="MC",color=:blue,legend=:bottom,yticks=:none,xlabel="\n kR")
    # Image
    scatter(ka,imag.(F0_EF),label="EF",xlabel="ka",color=:red,legend=:bottomright,title=L"$\mathfrak{\Im\! m}(\mathfrak{F}_0)$"*"\n\n")
    pI = plot!(twiny(),kR,imag.(F),grid=false,ribbon=σ_i,fillalpha=.3,label="MC",color=:blue,legend=:bottom,yticks=:none,xlabel="\n kR")
    plot(pR,pI,size = (750,400),ms=3)
     savefig("test1.pdf")



    ## test 2
    ## multipole decomposition for a plane wave small frequency
    ω = .1
    ϕ = 0.05 # 
    particle = Particle(Acoustic(2; ρ=0.3, c=0.3),Circle(1.0))
    sp_MC, sp_EF = generate_species(radius_big_cylinder,particle,ϕ,separation_ratio)


    basis_order = 10
    basis_field_order = 10
    kws_MC = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
        ,:nb_iterations_max=> 500
        ,:nb_iterations_step=> 100
        ,:prec=>1e-3
    )

    kws_EF = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
    )

    F_MC = sample_effective_t_matrix(ω, host_medium, sp_MC;kws_MC...);


    kstar, wavemode = effective_sphere_wavenumber(ω,[sp_EF],host_medium;
    radius_big_cylinder=20.0,basis_order=basis_order);
    N,D = t_matrix_num_denom(kstar,wavemode;basis_field_order=basis_field_order);
    T =  - vec(sum(N,dims=1)./sum(D,dims=1))
    T0 = - N[1+basis_order,:]./D[1+basis_order,:]


    scatter(-basis_field_order:basis_field_order
    ,real.(F_MC),markershape=:circle,label="MC",title=L"$\mathfrak{\Re\! e}(\mathfrak{F}_0)$"*"\n\n");

    scatter!(-basis_field_order:basis_field_order,
    real.(T),label="EF",markershape=:star);
    
    scatter!(-basis_field_order:basis_field_order,
    real.(T0),label="EF0",markershape=:cross)

    
   scatter(-basis_field_order:basis_field_order,
    imag.(F_MC),markershape=:circle,label="MC",title=L"$\mathfrak{\Im\! m}(\mathfrak{F}_0)$"*"\n\n");
  
    scatter!(-basis_field_order:basis_field_order,
    imag.(T),label="EF",markershape=:star);
    
    si = scatter!(-basis_field_order:basis_field_order,
    imag.(T0),label="EF0",markershape=:cross)
 

    plot(sr,si,title="k="*string(ω))

    savefig("k0.2.pdf")



     ## test 3
    ## multipole decomposition for a plane wave small frequency
    ω = 2.0
    ϕ = 0.05 # 
    particle = Particle(Acoustic(2; ρ=Inf, c=Inf),Circle(1.0))
    sp_MC, sp_EF = generate_species(radius_big_cylinder,particle,ϕ,separation_ratio)


    basis_order = 5
    basis_field_order = 5
    kws_MC = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
        ,:nb_iterations_max=> 10000
        ,:nb_iterations_step=> 1000
        ,:prec=>1e-2
    )

    kws_EF = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
    )

    F_MC = sample_effective_t_matrix(ω, host_medium, sp_MC;kws_MC...);


    kstar, wavemode = effective_sphere_wavenumber(ω,[sp_EF],host_medium;
    radius_big_cylinder=20.0,basis_order=basis_order);
    N,D = t_matrix_num_denom(kstar,wavemode;basis_field_order=basis_field_order);
    T =  - vec(sum(N,dims=1)./sum(D,dims=1))
    T0 = - N[1+basis_order,:]./D[1+basis_order,:]

    scatter(-basis_field_order:basis_field_order
    ,real.(F_MC),markershape=:circle,label="MC",title=L"$\mathfrak{\Re\! e}(\mathfrak{F}_0)$"*"\n\n");

    scatter!(-basis_field_order:basis_field_order,
    real.(T),label="EF",markershape=:star);
    
    sr = scatter!(-basis_field_order:basis_field_order,
    real.(T0),label="EF0",markershape=:cross);

    
   scatter(-basis_field_order:basis_field_order,
    imag.(F_MC),markershape=:circle,label="MC",title=L"$\mathfrak{\Im\! m}(\mathfrak{F}_0)$"*"\n\n");
  
    scatter!(-basis_field_order:basis_field_order,
    imag.(T),label="EF",markershape=:star);
    
    si = scatter!(-basis_field_order:basis_field_order,
    imag.(T0),label="EF0",markershape=:cross);
 

    plot(sr,si,title="k="*string(ω))

    @userplot PlotModes
    @recipe function f(pm::PlotModes)
        n, T, T0, F = pm.args
        # n = length(x)
        # inds = circshift(1:n, 1 - i)
        # linewidth --> range(0, 10, length = n)
        # seriesalpha --> range(0, 1, length = n)
        # aspect_ratio --> 1
        label --> false
        real.(T)
        real.(F)
    end

anim = @animate for i ∈ 1:1000
    plotmodes(basis_field_order,T,T0,[mean(F_MC[n][1:min(length(F_MC[n]),i)]) for n = 1:2*basis_field_order+1])
end

gif(anim, fps = 15)

    anim = @animate for i=10000
        plot(rand(i))
    end

    gif(anim,"anim1.gif",fps=7)


        f=[mean(F_MC[n][1:min(length(F_MC[n]),i)]) for n = 1:2*basis_field_order+1]
        scatter(-basis_field_order:basis_field_order,
        real.(T),label="EF",markershape=:cross);
        
        scatter!(-basis_field_order:basis_field_order,
        real.(T0),label="EF0",markershape=:circle);
        
        scatter!(-basis_field_order:basis_field_order
        ,real.(f),markershape=:x,label="MC",title=L"$\mathfrak{\Re\! e}(\mathfrak{F}_0)$"*"\n\n");

    
        
        scatter(-basis_field_order:basis_field_order,
        imag.(T),label="EF",markershape=:cross);
        
        scatter!(-basis_field_order:basis_field_order,
        imag.(T0),label="EF0",markershape=:circle);
        
        scatter!(-basis_field_order:basis_field_order,
        imag.(f),ylims=(0.0,0.14),markershape=:x,label="MC",title=L"$\mathfrak{\Im\! m}(\mathfrak{F}_0)$"*"\n\n");
      
    # end

    



    savefig("test2a.pdf")
    

    scatter(-basis_field_order:basis_field_order,abs.(F_EF-F_MC.μ))
    scatter!(title="Relative error")
    savefig("test2b.pdf")
    
    ## test 3
    ## multipole decomposition for a plane wave small frequency
    ω = .7
    ϕ = 0.05 # 
    particle = Particle(Acoustic(2; ρ=Inf, c=Inf),Circle(1.0))
    sp_MC, sp_EF = generate_species(radius_big_cylinder,particle,ϕ,separation_ratio)
    source = psource


    basis_order = 25
    basis_field_order = 25
    kws_MC = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
        ,:nb_iterations_max=> 50000
        ,:nb_iterations_step=> 1000
        ,:prec=>1e-2
    )

    kws_EF = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=> basis_order
        ,:basis_field_order=> basis_field_order
    )

    F_MC = averaged_multipole_decomposition(ω,sp_MC,source;kws_MC...)

    df = df_output(F_MC)
    CSV.write("test3.csv",df)

    F_EF = effective_multipole_decomposition(ω,[sp_EF],source;kws_EF...)
    
    ka = outer_radius(sp_MC)/real(host_medium.c) * ω
    kR = radius_big_cylinder/real(host_medium.c) * ω
     
    sr = scatter(-basis_field_order:basis_field_order,real.(F_EF),label="EF");
    sr = scatter!(-basis_field_order:basis_field_order,real.(F_MC.μ),label="MC",title=L"$\mathfrak{\Re\! e}(\mathfrak{F}_0)$"*"\n\n");
    si = scatter(-basis_field_order:basis_field_order,imag.(F_EF),label="EF");
    si = scatter!(-basis_field_order:basis_field_order,imag.(F_MC.μ),label="MC",title=L"$\mathfrak{\Im\! m}(\mathfrak{F}_0)$"*"\n\n");
    plot(sr,si)
    savefig("test3a.pdf")
    
    plot(sr,si,xlims=(-2.5,2.5))

    scatter(-basis_field_order:basis_field_order,abs.(F_EF-F_MC.μ)./abs.(F_MC.μ))
    scatter!(title="Relative error")
    savefig("test3b.pdf")

    scatter(-basis_field_order:basis_field_order,abs.(F_EF-F_MC.μ))
    scatter!(title="absolute error")
    savefig("test3c.pdf")


    
    radius_big_cylinder = 20.0
    basis_order = 5
    basis_field_order = 5
    ω = 0.2

    kws_MC = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=>basis_order
        ,:basis_field_order=> basis_field_order
        ,:nb_iterations_max=>5000
        ,:nb_iterations_step=>500
        ,:prec=>1e-1
    )

    # compare boost
    F_MC_boost,nb_particles = boost_average_multipole_decomposition(ω,sp_MC,source;
    radius_big_cylinder=radius_big_cylinder, basis_order=basis_order, nb_iterations=500);
    
    F_MC = averaged_multipole_decomposition(ω,sp_MC,source;kws_MC...);

    # compare effective waves

    r = particle_radius
    ϕ = F_MC.mean_nb_particles*r^2/(radius_big_cylinder-r)^2
    sp_EF = Specie(Acoustic(2; ρ=sp_MC.particle.medium.ρ, c=sp_MC.particle.medium.c),Circle(particle_radius);
    volume_fraction = ϕ,separation_ratio=separation_ratio)

    kws_EF = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=>basis_order
        ,:basis_field_order=>basis_field_order
    )
    F_EF = effective_multipole_decomposition(ω,[sp_EF],source;kws_EF...)


    # compare effective waves and  boost

    sr = scatter(-basis_field_order:basis_field_order,real.(F_MC.μ),label="MC");
    sr = scatter!(-basis_field_order:basis_field_order,real.(F_MC_boost),label="MC_boost");
    sr = scatter!(-basis_field_order:basis_field_order,real.(F_EF),label="EF");
    
    si = scatter(-basis_field_order:basis_field_order,imag.(F_MC.μ),label="MC");
    si = scatter!(-basis_field_order:basis_field_order,imag.(F_MC_boost),label="MC_boost");
    si = scatter!(-basis_field_order:basis_field_order,imag.(F_EF),label="EF");
    plot(sr,si)



    ## how far to go depending on ω?
    basis_order = 30
    basis_field_order = 30
    ω = 1.0

    kws_EF = Dict(
        :radius_big_cylinder=>radius_big_cylinder
        ,:basis_order=>basis_order
        ,:basis_field_order=>basis_field_order
    )
    F_EF = effective_multipole_decomposition(ω,[sp_EF],source;kws_EF...)

    sr = scatter(-basis_field_order:basis_field_order,real.(F_EF));
    si = scatter(-basis_field_order:basis_field_order,imag.(F_EF));
    plot(sr,si)