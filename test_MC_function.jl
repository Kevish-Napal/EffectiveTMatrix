basis_order=1
basis_field_order =1
particle_radius=1.0
multipole_decomposition_order = basis_field_order
ω=1.0
k=ω/host_c

dimension=2;
host_c=1.0;
host_medium = Acoustic(dimension; ρ=1.0, c=host_c);

x=[1.0,2.0]
translation_matrix = regular_translation_matrix(host_medium, basis_order, basis_field_order, ω, x)

function compute_multipole_decomposition(particles::AbstractParticles{2},source::AbstractSource{P};
    basis_order=10::Int, basis_field_order=0::Int) where P
    
    k = ω/source.medium.c
    
    function kernel_V(particles)
        V = complex(zeros(2*(basis_order+basis_field_order)+1,length(particles)))
        f_V = regular_basis_function(k,source.medium)
        for j = 1:length(particles)
            V[:,j] = f_V(basis_order+basis_field_order,-origin(particles[j]))
        end
    return V
    end

    sim = FrequencySimulation(particles,source);
    scattering_coefficients = basis_coefficients(sim, ω; basis_order = basis_order)
    V = kernel_V(particles)
    return [sum(V[n:n+2*basis_order,:].*scattering_coefficients) for n = 1+2*basis_field_order:-1:1] # n is in reverse order
 end

 function compute_multipole_decomposition_matriciel(particles::AbstractParticles{2},source::AbstractSource{P};
    basis_order=10::Int, basis_field_order=0::Int) where P
    
    nb_particles = length(particles)
    sim = FrequencySimulation(particles,source);
    scattering_coefficients = basis_coefficients(sim, ω; basis_order = basis_order)
    return sum(
        conj.(regular_translation_matrix(host_medium, basis_order, basis_field_order, ω, origin(particles[i])))*
        scattering_coefficients[:,i] for i=1:nb_particles)
 end



function multipole_decomposition_optimized(ω::Number, sp::Specie,source::AbstractSource{P};
    radius_big_cylinder=10.0::Float64, basis_order=10::Int, nb_iterations=10::Int,
    basis_field_order=0::Int) where P  

    function renew_particle_configurations()
        config = random_particles(
            sp.particle.medium,
            sp.particle.shape;
            region_shape = Circle(1.05*radius_big_cylinder),
            volume_fraction = sp.volume_fraction
        );
        config = config[norm.(origin.(config)) .< radius_big_cylinder .- outer_radius.(config)]
    end
    
    k = ω/source.medium.c


    F = complex(zeros(2*basis_field_order+1,nb_iterations))
    nb_particles = zeros(nb_iterations)
    
    for config=1:nb_iterations
        particles = renew_particle_configurations()
        nb_particles[config] = length(particles)
        sim = FrequencySimulation(particles,source);
        scattering_coefficients = basis_coefficients(sim, ω; basis_order = basis_order)
        F[:,config] = sum(conj.(regular_translation_matrix(host_medium, basis_order, basis_field_order, ω, origin(particles[i])))*scattering_coefficients[:,i] for i=1:length(particles))
    end

    return F, nb_particles
end

function kernel_V(particles)
    V = complex(zeros(2*(basis_order+multipole_decomposition_order)+1,length(particles)))
    f_V = regular_basis_function(k,host_medium)
    for j = 1:length(particles)
        V[:,j] = f_V(basis_order+basis_field_order,-origin(particles[j]))
    end
    return V
end

fn = rand(2*basis_order+1)+im*rand(2*basis_order+1)

particle = Particle(Acoustic(2; ρ=Inf, c=Inf),Circle(x,particle_radius))
V = kernel_V([particle])

@time begin 
    V = kernel_V([particle])
    F1 = [sum(V[n:n+2*basis_order].*fn)  for n = 1+2*multipole_decomposition_order:-1:1]
end


@time begin
    translation_matrix = regular_translation_matrix(host_medium, basis_order, basis_field_order, ω, x)
    F2 = conj.(translation_matrix)*fn
end

regular_basis_function(k,host_medium)(2,-x)

norm(F1-F2)


@time multipole_decomposition(ω, sp_MC,source; basis_field_order=0)
@time multipole_decomposition_optimized(ω, sp_MC,source; basis_field_order=0)

particles = renew_particle_configurations()
nb_particles=length(particles)

@time compute_multipole_decomposition(particles,psource)
@time compute_multipole_decomposition_matriciel(particles,psource)