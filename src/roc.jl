using Statistics

export roc, auc, auc_roc, confusion_matrix, confusion_matrix_nt

"""
	roc(pred_scores, real_scores)

Returns a tuple of vectors (FPR, TPR), where FPR = "False positive rate" and
TPR = "True positive rate".
Each score in `real_scores` is interpreted as positive if the score is positive.
To plot the ROC curve using Plots.jl: `plot(FPR, TPR, linetype=:steppost)`.
"""
function roc(pred_scores::AbstractVector, real_scores::AbstractVector)
	@assert length(pred_scores) == length(real_scores)
	perm = sortperm(pred_scores; rev=true)
	real = real_scores[perm]
    pred = pred_scores[perm]

	tpr = [0; cumsum(real .> 0)] ./ sum(real .> 0)
	fpr = [0; cumsum(real .≤ 0)] ./ sum(real .≤ 0)

    # remove extra points
    _jumps = [true; pred[2:end] .≠ pred[1:end-1]; true]

    return (fpr[_jumps], tpr[_jumps])
end

"""
	auc(pred_scores, real_scores)

Area under the ROC curve.
"""
function auc(pred_scores::AbstractVector, real_scores::AbstractVector)
	@assert length(pred_scores) == length(real_scores)
	fpr, tpr = roc(pred_scores, real_scores)
	auc_roc(fpr, tpr)
end

"""
	auc_roc(fpr, tpr)

Area under the ROC curve, from pre-computed FPR and TPR values.
"""
function auc_roc(fpr::AbstractVector, tpr::AbstractVector)
	@assert length(fpr) == length(tpr)
	sum((fpr[2:end] .- fpr[1:end-1]) .* middle.(tpr[2:end], tpr[1:end-1]))
end

"""
	confusion_matrix(pred_scores, real_scores, thresh=0)

Confusion matrix:
	[(true positives)  (false positives)
	 (false negatives) (true negatives)]
"""
function confusion_matrix(pred_scores::AbstractVector, real_scores::AbstractVector; thresh=0)
	tp = sum((pred_scores .> thresh) .& (real_scores .> 0))
	fn = sum((pred_scores .≤ thresh) .& (real_scores .> 0))
	tn = sum((pred_scores .≤ thresh) .& (real_scores .≤ 0))
	fp = sum((pred_scores .> thresh) .& (real_scores .≤ 0))
	[tp fp;
	 fn tn]
end

"""
	confusion_matrix_nt(pred_scores, real_scores, thresh=0)

Returns a NamedTuple, `(tp, fn, tn, fp)`.
"""
function confusion_matrix_nt(pred_scores::AbstractVector, real_scores::AbstractVector; thresh=0)
	tp = sum((pred_scores .> thresh) .& (real_scores .> 0))
	fn = sum((pred_scores .≤ thresh) .& (real_scores .> 0))
	tn = sum((pred_scores .≤ thresh) .& (real_scores .≤ 0))
	fp = sum((pred_scores .> thresh) .& (real_scores .≤ 0))
    return (tp = tp, fn = fn, tn = tn, fp = fp)
end
