## Timing
using Random,
  SpecialFunctions,
  CSV,
  LinearAlgebra,
  ROC,
  Base.Threads,
  BenchmarkTools,
  Plots,
  DataFrames
using Statistics
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

  Random.seed!(seed)

  ## evaluate computational cost
  rk2time = Float64[]
  rk4time = Float64[]
  si2time = Float64[]

  si2(X, y, tau, 5, reltol, max_iter, 123, 0)
  rk2(X, y, tau, 5, reltol, max_iter, 123, 0)
  rk4(X, y, tau, 5, reltol, max_iter, 123, 0)

  for i = 1:10
    append!(si2time, @elapsed si2(X, y, tau, 5, reltol, max_iter, seed+i, 0))
    append!(rk2time, @elapsed rk2(X, y, tau, 5, reltol, max_iter, seed+i, 0))
    append!(rk4time, @elapsed rk4(X, y, tau, 5, reltol, max_iter, seed+i, 0))
  end

  print("RK2:", round(1000*mean(rk2time),digits=2), ", ", round(1000*std(rk2time),digits=2), "\n")
  print("RK4:", round(1000*mean(rk4time),digits=2), ", ", round(1000*std(rk4time),digits=2), "\n")
  print("SI2:", round(1000*mean(si2time),digits=2), ", ", round(1000*std(si2time),digits=2), "\n")

  include("./nag-logi-bt.jl")
  include("./si2-logi-bt.jl")

  max_iter = UInt64(10^(3))
  sig = Float64(6)  ## sigma parameter for SI2 is fixed

  # for backtraking
  rho=Float64(0.1)
  myC=Float64(0.0001)
  initTau=Float64(0.05)

  si2bttime = Float64[]
  nagbttime = Float64[]

  nagbt(X, y, reltol, max_iter, seed, myC, rho, initTau, true)
  si2bt(X, y, sig, reltol, max_iter, seed, myC, rho, initTau)

  for i = 1:10
    append!(
      si2bttime,
      @elapsed si2bt(X, y, sig, reltol, max_iter, seed+i, myC, rho, initTau)
    )
    append!(
      nagbttime,
      @elapsed nagbt(X, y, reltol, max_iter, seed+i, myC, rho, initTau, true)
    )
  end

  print("SI2BT:", round(1000*mean(si2bttime),digits=2), ", ", round(1000*std(si2bttime),digits=2), "\n")
  print("NAGBT:", round(1000*mean(nagbttime),digits=2), ", ", round(1000*std(nagbttime),digits=2), "\n")

end
