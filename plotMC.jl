using RecipesBase

@recipe function plot(MC::MonteCarloResult;field_apply=real)
    η = uncertainty(MC)
    @series begin
        label --> "μ"
        yerror --> field_apply(η)
        field_apply.(MC.μ)
    end

    @series begin
        label --> "μeff"
        field_apply.(MC.μeff)
    end

    @series begin
        label --> "μeff0"
        field_apply.(MC.μeff0)
    end
end
