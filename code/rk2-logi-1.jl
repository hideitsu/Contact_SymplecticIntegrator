# Optimization by 2nd order Runge Kutta method
function rk2(X,y,tau=Float64(0.01),sig=Float64(5),reltol=Float64(10^(-3)), max_iter=Int32(100),seed=Int16(123),th=Float64(0.0),ret=true)
    Random.seed!(seed)

    # initial position and momenta
    time=Float64(1)
    qq=q=rand(Float64,size(X)[2])./10;
    pp=p=zeros(Float64,length(q))
    valuesf=Float64[]
    append!(valuesf,fn(q,X,y))
    ## iterations
    for t=1:max_iter
        # --- flow (1/2)---
        k1p = - ((2sig+1)/time) * p - (sig^2) * (time^(sig-2) ) * grad1(q,X,y)
        k1q = p

        # --- flow (2/2)---
        qq = q + tau * k1q
        pp = p + tau * k1p

        k2p = -(2*sig +1)/(time+tau) * pp - (sig^2) * ((time + tau)^(sig-2)) * grad1(qq,X,y)
        k2q = pp

        # time update
        time = time + tau
        q = q + 0.5 * tau * (k1q+k2q)
        p = p + 0.5 * tau * (k1p+k2p)

        append!(valuesf,fn(q,X,y))
        if t>2
            if (abs(valuesf[t-1] - valuesf[t])/valuesf[t] < reltol)
                break
            end
            if isnan(valuesf[length(valuesf)])
                return q,valuesf[1:(length(valuesf)-1)]
            elseif valuesf[length(valuesf)] < th
                return q,valuesf
            end
        end
    end
    if ret==true
        return q,valuesf
    end
end
