# Optimization by 2nd order Symplectic integrator with backtraking

function si2bt(X,y,sig=Float64(5),reltol=Float64(10^(-3)), max_iter=Int32(100),seed=Int16(123),myC=Float64(10^-4),rho=Float64(0.5),initTau=Float64(.1))
    Random.seed!(seed)
    # initial position and momenta
    mytime=Float64(1)
    q=rand(Float64,size(X)[2])./10
    p=zeros(Float64,length(q));

    valuesf=Float64[]
    append!(valuesf,fn(q,X,y))
    ## iterations
    for t=1:max_iter
        ## Adaptive step size by backtraking
        function optS()
            tau=initTau
            for j in 1:10
                tmp_p=p
                tmp_q=q

                ## --- mytime-shift (1/2) ---
                tmp_mytime = mytime + 0.5 * tau;
                tmp_q= tmp_q + (tmp_p/sig) * ((tmp_mytime+0.5tau)^(-2sig)- tmp_mytime^(-2sig))/2
                # --- flow by H_V ---
                gq=grad1(tmp_q,X,y)
                tmp_p= tmp_p + (sig/3 * ((tmp_mytime+tau)^(3sig) - tmp_mytime^(3sig))) * gq
                # --- flow by H_K (2/2) ---
                tmp_q= tmp_q + (tmp_p/sig)*((tmp_mytime+0.5*tau)^(-2sig) - tmp_mytime^(-2sig))/2
                searchDirection=(tmp_p/sig)*((tmp_mytime+0.5*tau)^(-2sig) - tmp_mytime^(-2sig))/2
                # condition check
                if (fn(tmp_q,X,y) <= valuesf[t] + myC*tau * dot(gq,searchDirection))
                    return(tau)
                else
                    tau=rho*tau
                end
            end
            return(tau)
        end
        tau=optS()  # obtain adaptive step size

        ## --- time-shift (1/2) ---
        mytime = mytime + 0.5 * tau;

        # --- flow by H_K (1/2)---
        q= q + (p/sig) * ((mytime+0.5tau)^(-2sig)- mytime^(-2sig))/2
        # --- flow by H_V ---
        gq=grad1(q,X,y)
        p= p + (sig/3 * ((mytime+tau)^(3sig) - mytime^(3sig))) * gq
        # --- flow by H_K (2/2) ---
        q= q + (p/sig)*((mytime+0.5*tau)^(-2sig) - mytime^(-2sig))/2


        # --- time-shift (2/2) ---
        mytime=mytime + 0.5*tau

        # --- store gradient descent values ---
        append!(valuesf,fn(q,X,y))
        if t>2
            if (abs(valuesf[t-1] - valuesf[t])/valuesf[t] < reltol)
                break
            end
            if isnan(valuesf[length(valuesf)])
                return q,valuesf[1:(length(valuesf)-1)]
            end
        end
    end
    return q,valuesf
end
