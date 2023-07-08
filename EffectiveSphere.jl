

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

# T-matrix is - Matrix_Num/Matrix_Denom
function t_matrix(ω::T,effective_cylinder::Material,outer_medium::Acoustic{T,2};
    basis_order=10::Integer,basis_field_order=10::Integer,include_terms=basis_order::Int) where T <: AbstractFloat

    radius_big_cylinder = effective_cylinder.shape.radius
    micro = effective_cylinder.microstructure
    species = micro.species

    kstar, wavemode =  effective_sphere_wavenumber(ω,species,outer_medium ;
    basis_order= basis_order,radius_big_cylinder=radius_big_cylinder)

    N, D = t_matrix_num_denom(kstar,wavemode;basis_field_order = basis_field_order)

    return - vec(sum(N[basis_order+1-include_terms:basis_order+1+include_terms,:],dims=1) ./ sum(D[basis_order+1-include_terms:basis_order+1+include_terms,:],dims=1))
end


function effective_T_matrices(ω,host_medium,sps_EF;radius_big_cylinder,basis_order,basis_field_order)

    kstar, wavemode = effective_sphere_wavenumber(ω,sps_EF,host_medium;
                radius_big_cylinder=radius_big_cylinder, basis_order=basis_order);

    N,D = t_matrix_num_denom(kstar,wavemode;basis_field_order=basis_field_order);

    T =  (- vec(sum(N,dims=1)./sum(D,dims=1)))
    T0 = (- N[1+basis_order,:]./D[1+basis_order,:])

    return T, T0
end