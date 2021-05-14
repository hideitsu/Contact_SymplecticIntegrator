# Optimization by 4th order Runge Kutta method
# --- RK for Zhang ---
function ZhangRK(X,y,q,p,tau=Float64(0.01),sig=Int8(5),time=Float64(1))
    return -((2*sig + 1)/time) * p - (sig^2 )*(time^(sig-2)) * grad1(q,X,y)
end


function rk4(X,y,tau=Float64(0.01),sig=Float64(5),reltol=Float64(10^(-3)), max_iter=Int32(100),seed=Int16(123),th=Float64(0.0))
    Random.seed!(seed)

    # initial position and momenta
    time=Float64(1)
    qq=q=rand(Float64,size(X)[2])./10;
    pp=p=zeros(Float64,length(q))
    valuesf=Float64[]
    append!(valuesf,fn(q,X,y))
    ## iterations
    for t=1:max_iter
        # --- flow (1/4)---
        k1p = ZhangRK(X,y,q,p,tau,sig,time)
        k1q = p

        # --- flow (2/4) ---
        qq = q + 0.5 * tau * k1q
        pp = p + 0.5 * tau * k1p

        k2p = ZhangRK(X,y,q,p,tau,sig,time + 0.5 * tau)
        k2q = pp

        # --- flow (3/4) ---
        qq = q + 0.5 * tau * k2q
        pp = p + 0.5 * tau * k2p

        k3p = ZhangRK(X,y,qq,pp,tau,sig,time + 0.5 * tau)
        k3q = pp

        # --- flow (4/4) ---
        qq = q + tau * k3q
        pp = p + tau * k3p

        k4p = ZhangRK(X,y,qq,pp,tau,sig,time + 0.5 * tau)
        k4q = pp

        # --- time update ---
        time = time + tau
        q = q + tau * (k1q + 2 * k2q + 2 * k3q + k4q)/6
        p = p + tau * (k1p + 2 * k2p + 2 * k4p + k4p)/6

        # --- store gradient descent values ---
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
    return q,valuesf
end
