## parameters
    dimension=2;
    host_medium = Acoustic(dimension; ρ=1.0, c=1.0);
    ω2k(ω) = real(ω/host_medium.c)
    k2ω(k) = real(k*host_medium.c)

    # sources
    rsource = regular_spherical_source(host_medium, [1.0+0.0im];
    position = zeros(dimension), symmetry = RadialSymmetry{dimension}()
    );
    psource = plane_source(host_medium; direction = [1.0,0.0])

    function mode_source(N::Int,dim::Int=2)
        coeffs = complex(zeros(2abs(N)+1))
        N<0 ? coeffs[1] = 1.0+0.0im : coeffs[end] = 1.0+0.0im
        return regular_spherical_source(host_medium, coeffs; 
            position = zeros(dim),
            symmetry = (N==0) ? RadialSymmetry{dim}() : WithoutSymmetry{dim}()
            );
    end

    ### quick check of the function mode_source
    # mode = 3
    # ω = 1.5
    # f_source = mode_source(mode).field
    # test_mode_source(x) = abs(f_source([x[1],x[2]],ω) - besselj(mode,ω*norm(x))*exp(im*mode*atan(x[2]/x[1])))
    # X = [[x,y] for x=1.0:.1:4.0 for y = 1.0:.1:4.0]
    # maximum(test_mode_source.(X)) < 1e-10

    # sizes
    particle_radius = 1.0
    radius_big_cylinder = 20.0
    separation_ratio=1.001

    # micro = Microstructure(host_medium,sp_EF);
    # effective_cylinder = Material(Circle(radius_big_cylinder),micro);
