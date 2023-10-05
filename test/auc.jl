import ROCCurves
using Test: @test, @testset

@testset "ROC#7" begin
    # Example from https://github.com/diegozea/ROC.jl/issues/7
    scores = [1, 2, 3, 4, 6, 5, 7, 8, 9, 10]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    @test ROCCurves.auc(scores, labels) ≈ 0.96
end

@testset "#2" begin
    # https://github.com/cossio/ROCCurves.jl/issues/2
    ŷ = [0, 1, 1, 1, 1, 0, 1 ,0 ,1]
    y = [0, 1, 1, 1, 1, 0, 0 ,0 ,1]
    @test ROCCurves.auc(ŷ, y) ≈ 0.875
    (fpr, tpr) = ROCCurves.roc(ŷ, y)
    @test fpr ≈ [0.0, 0.25, 1.00]
    @test tpr ≈ [0.0, 1.00, 1.00]
end
