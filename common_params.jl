## parameters
    dimension=2;
    host_medium = Acoustic(dimension; ρ=1.0, c=1.0);

    # sources
    rsource = regular_spherical_source(host_medium, [1.0+0.0im];
    position = zeros(dimension), symmetry = RadialSymmetry{dimension}()
    );
    psource = plane_source(host_medium; direction = [1.0,0.0])

    # sizes
    particle_radius = 1.0
    radius_big_cylinder = 20.0
    separation_ratio=1.001


    ϕ = 0.05 # 
    particle = Particle(Acoustic(2; ρ=Inf, c=Inf),Circle(1.0))
    sp_MC, sp_EF = generate_species(radius_big_cylinder,particle,ϕ,separation_ratio)

    micro = Microstructure(host_medium,sp_EF);
    effective_cylinder = Material(Circle(radius_big_cylinder),micro);
