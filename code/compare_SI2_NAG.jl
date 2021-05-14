## Comparison between SI2 and NAG
using Distributed
using Random,
  SpecialFunctions,
  CSV,
  LinearAlgebra,
  ROC,
  Base.Threads,
  BenchmarkTools,
  Plots,
  DataFrames
import StatsFuns.logistic
using StatsBase

dnames = ["Sonar", "BreastCancer", "Diabetis", "HouseVotes", "MNIST"]

seed = UInt8(123)

for dID = 1:length(dnames)
  global dname = dnames[dID]
  print(dname, "\n")
  include("./common.jl")
  include("./nag-logi-bt.jl")
  include("./si2-logi-bt.jl")

  reltol = Float64(10^(-6))
  max_iter = UInt64(10^(3))
  sig = Float64(6)  ## sigma parameter for SI2 is fixed
  Random.seed!(seed)

  # for backtraking
  rho = Float64(0.1)
  myC = Float64(0.0001)
  initTau = Float64(0.05)


  ## SI2
  wsi2, vasi2 = si2bt(X, y, sig, reltol, max_iter, seed, myC, rho, initTau)

  print("iteration=", length(vasi2), "\n")
  res = logistic.(testX * wsi2)
  rres = roc(res, testLabel, true)
  print(round(AUC(rres),digits=2),"\n")

  wnag, vanag = nagbt(X, y, reltol, max_iter, seed, myC, rho, initTau, true)
  print("iteration=", length(vanag), "\n")
  res = logistic.(testX * wnag)
  rres = roc(res, testLabel, true)
  print(round(AUC(rres),digits=2),"\n")

  fontsize=15
  titl = "SI2 and NAG: " * dname
  leg1 = "SI2"
  leg2 = "NAG"
  p = plot(
    vasi2,
    xaxis = :log,
    yaxis = :log,
    label = leg1,
    title = titl,
    linewidth = 2,
    xtickfontsize = fontsize - 2,
    ytickfontsize = fontsize - 2,
    xguidefontsize = fontsize + 1,
    yguidefontsize = fontsize + 1,
    legendfontsize = fontsize - 3,
    titlefontsize = fontsize,
  )
  plot!(
    vanag,
    xaxis = :log,
    yaxis = :log,
    label = leg2,
    title = titl,
    linewidth = 2,
    xtickfontsize = fontsize - 2,
    ytickfontsize = fontsize - 2,
    xguidefontsize = fontsize + 1,
    yguidefontsize = fontsize + 1,
    legendfontsize = fontsize - 3,
    titlefontsize = fontsize,
  )
  xlabel!("# of iteration")
  ylabel!("objective function value")
  p
  savefig("SI2_NAG_" * dname * ".pdf")
end
