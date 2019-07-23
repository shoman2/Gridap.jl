module GeometryTestsAll

using Test

@testset "GridGraphs" begin include("GridGraphsTests.jl") end

#@testset "Geometry" begin include("GeometryTests.jl") end
#
#@testset "GridPortions" begin include("GridPortionsTests.jl") end
#
#@testset "BoundaryGrids" begin include("BoundaryGridsTests.jl") end
#
#@testset "SkeletonGrids" begin include("SkeletonGridsTests.jl") end

end # module
