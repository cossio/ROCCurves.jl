using Plots, ROCCurves

real_scores = rand((+1,-1), 1000)
pred_scores = real_scores .+ 1 .* randn.()
plt = plot(identity, xlim=(0,1), ylim=(0,1),
	xlabel="FPR", ylabel="TPR", label=nothing,
	legend=:bottomright, line=:dash, color=:black, width=2)

As = Float64[]

for σ = 0:10
	pred_scores = real_scores .+ σ .* randn.()
	fpr, tpr = roc(pred_scores, real_scores)
	A = round(auc_roc(fpr, tpr); digits=2)
	push!(As, A)
	plot!(plt, fpr, tpr, linetype=:steppost,
		label="std=$σ, auc=$A")
end

plt
