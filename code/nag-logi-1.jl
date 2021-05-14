# Optimization by Nesterov Accelerated Gradient
function nag(X,y,snag=Float32(0.1),reltol=Float64(10^(-3)), max_iter=Int32(100),seed=Int16(123),th=Float64(0.0),ret=true)
    Random.seed!(seed)

    # initial position and momenta
    p=q=rand(Float64,size(X)[2])./10;
    pp=qq=zeros(Float64,length(q));
    valuesf=Float64[]
    append!(valuesf,fn(q,X,y))
    ## iterations
    for t=1:max_iter

        # --- gradient descent stage ---
        qq=p-snag*grad1(p,X,y);
        # --- momentum stage ---
        mu=Float64(t/(t+3));
        pp=qq+mu*(qq-q);

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
            elseif valuesf[length(valuesf)] < th
                return q,valuesf
            end
        end
    end
    if ret==true
        return q,valuesf
    end
end
