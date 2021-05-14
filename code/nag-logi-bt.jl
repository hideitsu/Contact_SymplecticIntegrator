# Optimization by Nesterov Accelerated Gradient with backtraking and restarting
function nagbt(X,y,reltol=Float64(10^(-3)), max_iter=Int32(100),seed=Int16(123),myC=Float64(10^-4),rho=Float64(0.5),initTau=Float64(.1),restart=true)
    Random.seed!(seed)

    # initial position and momenta
    p=q=rand(Float64,size(X)[2])./10;
    pp=qq=zeros(Float64,length(q));
    valuesf=Float64[]
    append!(valuesf,fn(q,X,y))
    ## iterations
    mcount=0
    for t=1:max_iter
        #qOld=q
        mcount=mcount+1
        mu=Float64(mcount/(mcount+3));
        ## Adaptive step size by back tracking
        function optS()
            gp=grad1(p,X,y)
            tau=initTau
            for j in 1:10
                tmp_qq=qq
                tmp_q=q
                tmp_pp=pp
                tmp_p=p

                # --- gradient descent stage ---
                tmp_qq=tmp_p-tau*gp
                # --- momentum stage ---
                tmp_pp=tmp_qq+mu*(tmp_qq-tmp_q);
                tmp_q=tmp_qq
                tmp_p=tmp_pp

                searchDirection=-gp
                # condition check
                if (fn(tmp_q,X,y) <= valuesf[t] + myC*tau * dot(gp,searchDirection))
                    return(tau)
                else
                    tau=rho*tau
                end
            end
            return(tau)
        end
        tau=optS()
        # --- gradient descent stage ---
        qq=p-tau*grad1(p,X,y);
        # --- momentum stage ---
        pp=qq+mu*(qq-q);

        ## restart
        if restart
            if fn(q,X,y) > valuesf[length(valuesf)]
                break
            end
        end


        append!(valuesf,fn(q,X,y))
        # --- update ---
        q=qq
        p=pp
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
