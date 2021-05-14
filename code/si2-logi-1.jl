# Optimization by 2nd order Symplectic integrator

function si2(X,y,tau=Float64(0.0001),sig=Float64(5),reltol=Float64(10^(-3)), max_iter=Int32(100),seed=Int16(123),th=Float64(0.0))
    Random.seed!(seed)

    # initial position and momenta
    mytime=Float64(1)
    q=rand(Float64,size(X)[2])./10
    p=zeros(Float64,length(q));

    valuesf=Float64[]
    append!(valuesf,fn(q,X,y))
    ## iterations
    for t=1:max_iter
        ## --- time-shift (1/2) ---
        mytime = mytime + 0.5 * tau

        # --- flow by H_K (1/2)---
        q= q + (p/sig) * ((mytime+0.5tau)^(-2sig)- mytime^(-2sig))/2

        # --- flow by H_V ---
        gq=grad1(q,X,y)
        p= p + (sig/3 * ((mytime+tau)^(3sig) - mytime^(3sig))) * gq

        # --- flow by H_K (2/2) ---
        q= q + (p/sig)*((mytime+0.5*tau)^(-2sig) - mytime^(-2sig))/2

        # --- mytime-shift (2/2) ---
        mytime=mytime + 0.5*tau

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
