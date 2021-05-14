Random.seed!(123)
eps4log = Float64(10^(-14))

X=CSV.read("../data/"*dname*"/X.csv",DataFrame)
X=Matrix{Float64}(X)
y=CSV.read("../data/"*dname*"/y.csv",DataFrame)
y=Matrix{Float64}(y)
y=y[:,]
testX=CSV.read("../data/"*dname*"/testX.csv",DataFrame)
testX=Matrix{Float64}(testX)
testLabel=CSV.read("../data/"*dname*"/testLabel.csv",DataFrame)
testLabel=Matrix{Float64}(testLabel)
testLabel=testLabel[:,]


lambda=size(X)[2]*10^-8


# --- the function value ---
function fn(q,X,y,lambda=0.01)
    h = logistic.(X*q);
    if (findmax(abs.(h))[1]+eps4log>1) | (findmin(abs.(h))[1]<eps4log)
        return NaN
    else
        J = (y .* log.(abs.(h)) + (repeat([1],outer=length(y)) .- y) .* log.(abs.(1 .- h))) ./ length(y);

        return -sum(J) + lambda*norm(q)^2
    end
end

# --- the gradient vector ---
function grad1(q,X,y,lambda=0.01)
    h=logistic.(X*q);
    gr=(transpose(X) * ( h .- y)) ./ length(y) + 2*lambda*q
    return gr
end
