import Aqua
import ROCCurves
using Test: @testset

@testset verbose = true "aqua" begin
    Aqua.test_all(ROCCurves)
end
