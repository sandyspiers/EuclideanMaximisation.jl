using Test
using EuclideanMaximisation.Model
using EuclideanMaximisation.Generator
using EuclideanMaximisation.Solvers
import JuMP

@testset "Model.jl" begin
    @testset "Euclidean distance matrix" begin
        # test euclidean distance matrix calculations
        locations = [1 1; 2 0]
        # each row is a location
        edm = build_edm(locations, true)
        n1, n2 = size(edm)
        @test n1 == n2 == 2
        @test edm[1, 1] == zero(edm[1, 1])
        @test edm[1, 2] == edm[2, 1] ≈ sqrt(2)

        # col is a location
        edm = build_edm(locations, false)
        n1, n2 = size(edm)
        @test n1 == n2 == 2
        @test edm[1, 1] == zero(edm[1, 1])
        @test edm[1, 2] == edm[2, 1] == 2.0

        # check edm is valid
        @test Model.check_edm_valid(edm)
        @test Model.check_edm_valid(build_edm(rand(0:100, (25, 25))))
        @test !Model.check_edm_valid(rand(-100:100, (25, 25)))
        tmp_edm = rand(-100:100, (25, 25))
        tmp_edm += tmp_edm'
        @test !Model.check_edm_valid(tmp_edm)
    end

    @testset "EmsModel" begin
        # Check we can construct a valid model
        mdl = JuMP.Model()
        # Add some loc-vars
        n = 10
        x = JuMP.@variable(mdl, x[1:n] >= 0)
        # Create a distance matrix of n locations
        locations = rand(0:100, (n, 2))
        edm = build_edm(locations)
        @test size(edm) == (n, n)
        ems = EmsModel(; mdl = mdl, loc_dvars = x, edm = edm)
        @test Model.check_model_valid(ems)

        # Test fail on nonpositive loc dvars
        JuMP.set_lower_bound(x[1], -1)
        @test !Model.check_model_valid(ems)
    end
end

@testset "Generators.jl" begin
    @testset "random generators" begin
        @test Model.check_model_valid(
            random_capacitated_diversity_problem(25, 2, 0.5),
        )
        @test Model.check_model_valid(
            random_generalized_diversity_problem(1000, 2, 0.2, 0.5),
        )
        @test Model.check_model_valid(
            random_bilevel_diversity_problem(1000, 2, 0.2, 0.1),
        )
        @test Model.check_model_valid(random_diversity_problem(25, 2, 0.5))
    end
    @testset "file readers" begin
        # CDP
        cdp = file_capacitated_diversity_problem(
            "instances/GKD-b_11_n50_b02_m5.txt",
        )
        @test size(cdp.edm) == (50, 50)
        @test cdp.edm[1, 3] == 157.4
        @test length(cdp.loc_dvars) == 50
        @test Model.check_model_valid(cdp)
        # GDP
        gdp = file_generalized_diversity_problem(
            "instances/GKD-b_11_n50_b02_m5_k02.txt",
        )
        @test size(gdp.edm) == (50, 50)
        @test gdp.edm[1, 2] == 124.5
        @test length(gdp.loc_dvars) == 50
        @test Model.check_model_valid(gdp)
        # DP
        dp = file_diversity_problem("instances/GKD_d_1_n25_coor.txt", 0.2)
        @test size(dp.edm) == (25, 25)
        @test dp.edm[1, 2] == 32.34843
        @test length(dp.loc_dvars) == 25
        @test Model.check_model_valid(dp)
    end
end

@testset "Solvers.jl" begin
    @testset "Solvers Converge" begin
        solvers = AVALIABLE_SOLVERS
        for solver in solvers
            ems = random_capacitated_diversity_problem(25, 2, 0.5, 1)
            JuMP.set_silent(ems.mdl)
            @test solve!(ems, solver).obj_value > 0
        end
    end

    @testset "Solvers Agree" begin
        ems = random_capacitated_diversity_problem(20, 2, 0.5, 1)
        JuMP.set_silent(ems.mdl)
        solvers = AVALIABLE_SOLVERS
        obj_vals = []
        for solver in solvers
            _cpy = Model.copy(ems)
            JuMP.set_silent(_cpy.mdl)
            push!(obj_vals, solve!(_cpy, solver).obj_value)
        end
        for i in 1:length(solvers)-1
            @test abs(obj_vals[i] - obj_vals[i+1]) <= 1e-6 || obj_vals
        end
    end

    @testset "Timeout gracefully" begin
        time_limit = 3
        solvers = ["repoa", "fcard"]
        for solver in solvers
            ems = random_diversity_problem(200, 20, 0.1, 1)
            JuMP.set_silent(ems.mdl)
            JuMP.set_time_limit_sec(ems.mdl, time_limit)
            result = solve!(ems, solver)
            @test result.obj_value > 0
            @test abs(result.run_time - time_limit) < 1.0 || result
        end
    end
end
