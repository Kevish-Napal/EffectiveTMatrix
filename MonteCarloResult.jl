
# We need two different structures to stoere the data:
# 1) "struct MonteCarloTemp" to save all the realisations for checkings in case of problem
# 2) "struct MonteCarlo" to save only the average of the realisations

# This  structure will contain all realisations
mutable struct MonteCarloResultTemp
    basis_order::Int64
    basis_field_order::Int64

    ω::Float64
    sp_MC::Specie
    R::Float64

    F::Vector{Vector{ComplexF64}}

    μeff::Vector{ComplexF64}
    μeff0::Vector{ComplexF64}

    function MonteCarloResultTemp(
        basis_order::Int64,
        basis_field_order::Int64,

        ω::Float64,
        sp_MC::Specie,
        R::Float64,

        F::Vector{Vector{ComplexF64}},

        μeff::Vector{ComplexF64},
        μeff0::Vector{ComplexF64})

        if length(F) != basis_field_order+1 || length(μeff) != basis_field_order+1 || length(μeff0) != basis_field_order+1
            error("F, μeff and μeff0 have to be of length basis_field_order+1")
        end

        new(basis_order,basis_field_order,ω,sp_MC,R,F,μeff,μeff0)
    end
end

mutable struct MonteCarloResult
    basis_order::Int64
    basis_field_order::Int64

    ω::Float64
    sp_MC::Specie
    R::Float64

    μ::Vector{ComplexF64} 
    σ::Vector{ComplexF64}
    nb_iterations::Vector{Int64}

    μeff::Vector{ComplexF64}
    μeff0::Vector{ComplexF64}

    function MonteCarloResult(
        basis_order::Int64,
        basis_field_order::Int64,

        ω::Float64,
        sp_MC::Specie,
        R::Float64,

        μ::Vector{ComplexF64},
        σ::Vector{ComplexF64},
        nb_iterations::Vector{Int64},

        μeff::Vector{ComplexF64},
        μeff0::Vector{ComplexF64})

        if length(μ) != basis_field_order+1 || length(σ) != basis_field_order+1 || length(nb_iterations)!= basis_field_order+1 || length(μeff) != basis_field_order+1 || length(μeff0) != basis_field_order+1
            error("μ, σ, nb_iterations, μeff and μeff0 have to be of length basis_field_order+1")
        end

        new(basis_order,basis_field_order,ω,sp_MC,R,μ,σ,nb_iterations,μeff,μeff0)
    end
end

function MonteCarloResult(MCtemp::MonteCarloResultTemp)
    μ = mean.(MCtemp.F)
    σ = [std(real.(MCtemp.F[N]);mean=real(μ[N])) + im*std(imag.(MCtemp.F[N]);mean=imag(μ[N])) for N=1:MCtemp.basis_field_order+1]
    nb_iterations = length.(MCtemp.F)
    return MonteCarloResult(MCtemp.basis_order,MCtemp.basis_field_order,MCtemp.ω, MCtemp.sp_MC, MCtemp.R ,μ , σ ,nb_iterations ,MCtemp.μeff ,MCtemp.μeff0)
end

function MonteCarloResult(MCtemp_vec::Vector{MonteCarloResultTemp})
    return [MonteCarloResult(MCtemp) for MCtemp in MCtemp_vec]
end

function uncertainty(V::Vector{Float64})
    return 1.96*std(V)/sqrt(length(V))    
end

function uncertainty(V::Vector{ComplexF64})
    return uncertainty(real.(V))+im*uncertainty(imag.(V))
end

function uncertainty(MC::MonteCarloResult)
    return 1.96*MC.σ./sqrt.(MC.nb_iterations)
end

function relative_error(MC::MonteCarloResult)
    return abs.(MC.μ-MC.μeff)./abs.(MC.μ), abs.(MC.μ-MC.μeff0)./abs.(MC.μ)    
end

function CSV_string_vec(v::Vector)
    string_v = string()
    for x in v
        string_v *= "$x,"
    end
    return "\""*chop(string_v)*"\""
end

function MC_write(MC_vec::Vector{MonteCarloResult},file_path::String)
    header = [
        "basis_order",
        "basis_field_order",

        "ω",
        "R",
        
        "particle_radius",
        "c_particle",
        "ρ_particle",
        "separation_ratio",
        "volume_fraction",

        "μ",
        "σ",
        "nb_iterations",

        "μeff",
        "μeff0"
        ]

        open(file_path, "w") do f
            H=string()
            for h in header
                H*=h*","
            end 
            write(f, chop(H))

            write(f,"\n")
            for MC in MC_vec
                write(f,
                    "$(MC.basis_order),",
                    "$(MC.basis_field_order),",

                    "$(MC.ω),",
                    "$(MC.R),",

                    "$(outer_radius(MC.sp_MC.particle)),",
                    "$(MC.sp_MC.particle.medium.c),",
                    "$(MC.sp_MC.particle.medium.ρ),",
                    "$(MC.sp_MC.separation_ratio),",
                    "$(MC.sp_MC.volume_fraction),"
                )

                write(f, CSV_string_vec(MC.μ) ,",")
                write(f, CSV_string_vec(MC.σ) ,",")
                write(f, CSV_string_vec(MC.nb_iterations) ,",")

                write(f, CSV_string_vec(MC.μeff) ,",")
                write(f, CSV_string_vec(MC.μeff0))

                write(f,"\n")
            end
            write(f,"\n")
        end
end

function MC_read(file_path::String)
    file = CSV.File(file_path; types=Dict(:c_particle => ComplexF64)) 
    MC_vec=Vector{MonteCarloResult}()
    for i = 1:length(file)
        particle = Particle(Acoustic(2; ρ=file.ρ_particle[i], c=file.c_particle[i]),Circle(file.particle_radius[i]))
        sp_MC = Specie(particle; volume_fraction = file.volume_fraction[i],separation_ratio=file.separation_ratio[i]) 
        
        μ = parse.(ComplexF64,split(file.μ[i],","))
        σ = parse.(ComplexF64,split(file.σ[i],","))
        nb_iterations = parse.(Int64,split(file.nb_iterations[i],","))

        μeff = parse.(ComplexF64,split(file.μeff[i],","))
        μeff0 = parse.(ComplexF64,split(file.μeff0[i],","))

        push!(MC_vec,
                MonteCarloResult(file.basis_order[i],file.basis_field_order[i],file.ω[i],sp_MC,file.R[i],μ,σ,nb_iterations,μeff,μeff0)
                )
    end
    return MC_vec
end


function MCtemp_write(MC_vec::Vector{MonteCarloResultTemp},file_path::String)
    header = [
        "basis_order",
        "basis_field_order",

        "ω",
        "R",
        
        "particle_radius",
        "c_particle",
        "ρ_particle",
        "separation_ratio",
        "volume_fraction",

        "μeff",
        "μeff0"
        ]

        open(file_path, "w") do f
            H=string()
            for h in header
                H*=h*","
            end 
            write(f, H)

            FH = string()
            for N = 0:basis_field_order
                FH*="F"*string(N)*","
            end
            write(f, chop(FH))

            write(f,"\n")
            for MC in MC_vec
                write(f,
                    "$(MC.basis_order),",
                    "$(MC.basis_field_order),",

                    "$(MC.ω),",
                    "$(MC.R),",

                    "$(outer_radius(MC.sp_MC.particle)),",
                    "$(MC.sp_MC.particle.medium.c),",
                    "$(MC.sp_MC.particle.medium.ρ),",
                    "$(MC.sp_MC.separation_ratio),",
                    "$(MC.sp_MC.volume_fraction),"
                )

                write(f, CSV_string_vec(MC.μeff) ,",")
                write(f, CSV_string_vec(MC.μeff0),",")
                
                for N = 1:basis_field_order
                    write(f, CSV_string_vec(MC.F[N]) ,",")
                end
                write(f, CSV_string_vec(MC.F[basis_field_order+1]))

                write(f,"\n")
            end
            write(f,"\n")
        end
end

function MCtemp_read(file_path::String)
    file = CSV.File(file_path; types=Dict(:c_particle => ComplexF64)) 
    MCtemp_vec=Vector{MonteCarloResultTemp}()
    for i = 1:length(file)
        particle = Particle(Acoustic(2; ρ=file.ρ_particle[i], c=file.c_particle[i]),Circle(file.particle_radius[i]))
        sp_MC = Specie(particle; volume_fraction = file.volume_fraction[i],separation_ratio=file.separation_ratio[i]) 

        μeff = parse.(ComplexF64,split(file.μeff[i],","))
        μeff0 = parse.(ComplexF64,split(file.μeff0[i],","))

        F = [ComplexF64[] for _ = 0:file.basis_field_order[i]]
        for N=0:file.basis_field_order[i]
            F[N+1] = parse.(ComplexF64,split(getproperty(file,Symbol("F"*string(N)))[i],","))
        end

        push!(MCtemp_vec,
                MonteCarloResultTemp(file.basis_order[i],file.basis_field_order[i],file.ω[i],sp_MC,file.R[i],F,μeff,μeff0)
                )
    end
    return MCtemp_vec
end


# please provide folder path which CONTAINS the folder named "Data" inside
function save(MC_vec::Vector{MonteCarloResult},description::String,all_data_path::String=pwd())

    # Check all_data_path is a valid path 
    if !isdir(all_data_path)
        all_data_path = pwd()
        @warn("path "* all_data_path * "does not exist and is replaced with "* pwd())
    end

    # chop last "/" in pathname if any
    if all_data_path[end] == '/'
        all_data_path=chop(all_data_path)
    end

    # check if all_data_path contains folder Data
    # create Data folder otherwise.
    if !isdir(all_data_path*"/Data")
    # folder Data not found
        if all_data_path[end-4:end]!="/Data" 
        # path doesn't include Data neither
            mkdir(all_data_path*"/Data")
            @warn("Data Folder created at "* all_data_path)
        else 
        # path includes Data
            all_data_path = chop(all_data_path;tail=5)
        end
    end

    # update metadata file if it exists
    metadata_path = all_data_path*"/Data/metadata.csv"
    if isfile(metadata_path)
    # metadata file exists: update it
        global num_current_data
        open(metadata_path) do f
            num_total_data = -1 
            for _ in enumerate(eachline(f))
                num_total_data += 1
            end
            global num_current_data = num_total_data+1
        end
        open(metadata_path,"a") do f
            write(f,"$num_current_data,",description,",",string(Dates.now()),"\n")
        end
    else
    # metadata file doesn't exist: create, put headers, and new data info
        open(metadata_path,"w") do f
            num_current_data = 1
            write(f,"Folder,Description,Date\n")
            write(f,"$num_current_data,",description,",",string(Dates.now()),"\n")
        end
    end    

    # save new data
    new_data_path = all_data_path*"/Data/"*string(num_current_data)
    if isdir(new_data_path)
        @error("Folder $new_data_path already exist.")
    end
    mkdir(new_data_path)
    MC_write(MC_vec,new_data_path*"/MC.csv")
    return new_data_path
end

function save(MCtemp_vec::Vector{MonteCarloResultTemp},description::String,all_data_path::String=pwd())
    new_data_path = save(MonteCarloResult(MCtemp_vec),description,all_data_path)
    realisations_path = new_data_path*"/realisations"
    mkdir(realisations_path)
    MCtemp_write(MCtemp_vec,realisations_path*"/MCtemp.csv")
end


function Base.show(io::IO,MC::MonteCarloResult)
    basis_field_order = length(MC.μ)-1
    stdm_r = real.(MC.σ)./ sqrt.(MC.nb_iterations)
    stdm_i = imag.(MC.σ)./ sqrt.(MC.nb_iterations)

    print(io,"----------------------------------------------------------------------------------------------------")
    print(io, 
        "
        ω = $(MC.ω)
        container radius: $(MC.R)
        Particle type: $(MC.sp_MC.particle) 
        Values:
     ")

     for N = 0:basis_field_order
        print(io, 
        "
        F_$(N) =  $(MC.μ[N+1]) ± $(stdm_r[N+1]+im*stdm_i[N+1]) ($(MC.nb_iterations[N+1]) iterations) 
        ")
     end

     print(io,"----------------------------------------------------------------------------------------------------")
end