using Test, ROCCurves, Statistics

real_scores = rand((+1,-1), 1000)
pred_scores = real_scores .+ 1 .* randn.()
pred_scores_sorted = sort(pred_scores; rev=true)
Ms = [confusion_matrix.(Ref(pred_scores), Ref(real_scores); thresh=θ)
      for θ in [pred_scores_sorted; -Inf]]
TPs = getindex.(Ms, 1, 1)
FPs = getindex.(Ms, 1, 2)
FNs = getindex.(Ms, 2, 1)
TNs = getindex.(Ms, 2, 2)
@test all(TPs .+ FNs .== sum(real_scores .> 0))
@test all(FPs .+ TNs .== sum(real_scores .≤ 0))
fpr, tpr = roc(pred_scores, real_scores)
@test length(fpr) == length(tpr) == length(real_scores) + 1
@test fpr[1] == tpr[1] == 0
@test fpr[end] == tpr[end] == 1
@test tpr ≈ TPs ./ sum(real_scores .> 0)
@test fpr ≈ FPs ./ sum(real_scores .≤ 0)

pred_scores = real_scores
@test auc(pred_scores, real_scores) ≈ 1

pred_scores = -real_scores
@test auc(pred_scores, real_scores) ≈ 0

M  = confusion_matrix(pred_scores, real_scores)
nt = confusion_matrix_nt(pred_scores, real_scores)
@test M == [nt.tp nt.fp;
            nt.fn nt.tn]
