## Comparison between SI2, RK2/4, with respect to different sigma
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
  include("./rk2-logi-1.jl")
  include("./rk4-logi-1.jl")
  include("./si2-logi-1.jl")

  reltol = Float64(10^(-6))
  max_iter = UInt64(10^(10))
  tau = Float64(0.01)

  Random.seed!(123)
  sigRange = [2, 4, 5, 8, 10, 12]

  lt = [:dash, :dot, :dashdot, :dashdotdot, :solid]
  lt = vcat(lt, lt)
  lt = vcat(lt, lt)

  # number of threads
  nth = 5

  fontsize = 13

  ## SI2
  vas = []
  @threads for i = 1:length(sigRange)
    w, va = si2(X, y, tau, sigRange[i], reltol, max_iter)
    push!(vas, va)
  end

  titl = "SI2:" * dname
  leg1 = "sigma=" * string(round(sigRange[1], digits = 2))
  p = plot(
    vas[1],
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

  for i = 2:length(sigRange)
    leg = "sigma=" * string(round(sigRange[i], digits = 2))
    plot!(
      p,
      vas[i],
      xaxis = :log,
      yaxis = :log,
      label = leg,
      linewidth = 2,
      linestyle = lt[i],
    )
  end
  xlabel!("# of iteration")
  ylabel!("objective function value")
  p
  savefig("SI2_" * dname * "_l2.pdf")


  ## RK2
  vas = []
  @threads for i = 1:length(sigRange)
    w, va = rk2(X, y, tau, sigRange[i], reltol, max_iter)
    push!(vas, va)
  end

  titl = "RK2:" * dname
  leg1 = "sigma=" * string(round(sigRange[1], digits = 2))
  p = plot(
    vas[1],
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
  for i = 2:length(sigRange)
    leg = "sigma=" * string(round(sigRange[i], digits = 2))
    plot!(
      p,
      vas[i],
      xaxis = :log,
      yaxis = :log,
      label = leg,
      linewidth = 2,
      linestyle = lt[i],
    )
  end
  xlabel!("# of iteration")
  ylabel!("objective function value")
  p
  savefig("RK2_" * dname * "_l2.pdf")

  ## RK4
  vas = []
  @threads for i = 1:length(sigRange)
    w, va = rk4(X, y, tau, sigRange[i], reltol, max_iter)
    push!(vas, va)
  end

  titl = "RK4:" * dname
  leg1 = "sigma=" * string(round(sigRange[1], digits = 2))
  p = plot(
    vas[1],
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
  for i = 2:length(sigRange)
    leg = "sigma=" * string(round(sigRange[i], digits = 2))
    plot!(
      p,
      vas[i],
      xaxis = :log,
      yaxis = :log,
      label = leg,
      linewidth = 2,
      linestyle = lt[i],
    )
  end
  xlabel!("# of iteration")
  ylabel!("objective function value")
  p
  savefig("RK4_" * dname * ".pdf")
end
