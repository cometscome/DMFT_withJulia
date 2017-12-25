#This is a test code.
#Sparse modeling approach to analytical continuation of imaginary-time quantum Monte Carlo data
# Please see, J. Otsuki et al., Phys. Rev. E 95, 061302(R) (2017).
#                                10/18/2017(MM/DD/YYYY): This code was made by Yuki Nagai, Ph.D 

module Smac
    export smac_main!

    function smac_main!(M,N,orbitals,omegamax,β,vec_G)
        mat_K = zeros(Float64,M,N)
        L = min(M,N)
        #println(L)
    
        println("Making the matrix K...")
        mat_K = calc_K!(M,N,omegamax,β,mat_K)
        println("done.")
        
        println("Doing SVD...")
        (U,S,V) = svd(mat_K)
        println("done.")
    
        ii = 0
        for i in 1:L
            ii += 1
            if S[i] < 1e-10                
                break
            end
        end
        L = ii  
        println("---------------------------------------------")
        println("Singular values")
        for i in 1:L
            println(i,"\t",S[i])
        end
        println("---------------------------------------------")
        
        #println(L)
        vec_S = zeros(Float64,L)
        vec_S[1:L] = S[1:L]
        mat_U = zeros(Float64,M,L)
        mat_U[1:M,1:L] = U[1:M,1:L]
        #println(mat_U)
            
        mat_V = zeros(Float64,N,L)
        mat_V[1:N,1:L] = V[1:N,1:L]
        #println(mat_Vt)
    
        ξ = zeros(Float64,L)
        e = fill(1.0,N)
       
        ξ = mat_V'*e
        #println(length(ξ))
       
        vec_Gout = zeros(Float64,M,orbitals)
        xout = zeros(Float64,N,orbitals)
    
        for i in 1:orbitals
            println("\n","orbital: ",i)
            xout[:,i] = smac_est!(vec_G[:,i,1],M,N,mat_K,mat_U,vec_S,mat_V,ξ)           
            vec_Gout[:,i] = mat_K*xout[:,i]
            println("---------------------------------------------")
        end
    
    
        return xout,vec_Gout
        
    
    end

    function smac_est!(y,M,N,mat_K,mat_U,vec_S,mat_V,ξ)
    
        μ=1.0
        μp = 1.0
        fchiold = 0
        λest = 0
    
        l0 = -6.0
        lam0 =  10.0^(l0)
        λ =lam0
        itemax = 100000
        #xout = zeros(Float64,M)

        xout = smac_calc!(y,N,itemax,mat_U,vec_S,mat_V,ξ,λ,μ,μp)
        χ0 = dot(y-mat_K*xout,y-mat_K*xout)
        #println("χ0 ",χ0)
        lchi0 = log10(χ0)
    
        l1 = 1.0
        lam1 =  10.0^(l1)
        λ =lam1
        xout = smac_calc!(y,N,itemax,mat_U,vec_S,mat_V,ξ,λ,μ,μp)
        χ1 = dot(y-mat_K*xout,y-mat_K*xout)
        #println("χ1 ",χ1)
        lchi1 = log10(χ1)
    
        b = (log(χ0)-log(χ1))/(log(lam0)-log(lam1))
        a = exp(log(χ0) - b*log(lam0))
    
        for il in 2:20-1
            ll = (il-1)*(l1-l0)/(20-1)+ l0
            #println("ll ",ll)
            λ = 10^(ll)
            xout = smac_calc!(y,N,itemax,mat_U,vec_S,mat_V,ξ,λ,μ,μp)
            χ1 = dot(y-mat_K*xout,y-mat_K*xout)
            fchi = a*λ^b/χ1
            println(il-1,"/","18\t","λ:",λ,"\t\t", "Error:",χ1, "\t")
            #println(il,"/","19\t",λ,"\t", χ1, "\t", fchi)
            if il > 3
                if fchi > fchiold
                    λest = λ
                    fchiold = fchi
                end
            else
                fchiold = fchi
            end
        end
    
        λ = λest
        itemax = 1000000
        println("Appropriate λ: ",λ)
        println("Calculating final SMAC...")
        xout = smac_calc!(y,N,itemax,mat_U,vec_S,mat_V,ξ,λ,μ,μp)
        println("Done.")
        return xout
    end

    function smac_calc!(y,N,itemax,mat_U,vec_S,mat_V,ξ,λ,μ,μp)
        L = length(vec_S)
        yp = zeros(Float64,L)
        yp = mat_U'*y
        #println(sum(y))
    
    
        styp = zeros(Float64,L)
        stxp = zeros(Float64,L)
    
        xp = zeros(Float64,L)
        zp = zeros(Float64,L)
        up = zeros(Float64,L)
    
        z = zeros(Float64,N)
        u = zeros(Float64,N)
    
        for i in 1:L
            styp[i] = vec_S[i]*yp[i]
        end
    
        #println(sum(styp))

        for ite in 1:itemax
            (xp,zp,up,z,u)=smac_updates!(styp,xp,zp,up,z,u,vec_S,mat_V,ξ,λ,μ,μp) 
            #println(sum(xp))
            
        
            for i in 1:L
                stxp[i] = vec_S[i]*xp[i]
            end
            #hi = dot(yp-stxp,yp-stxp)
            #println(ite," ", sum(abs.(z - mat_V*xp)))
            #if ite % (itemax/10) == 0 
            #    println(ite,"\t", sum(abs.(z - mat_V*xp)))
            #end
            if sum(abs.(z - mat_V*xp)) < 1e-8
                break
            end
        end
    
        #xout = zeros(Float64,M)
        xout = mat_V*xp
        
    
        return xout
    end

    function smac_updates!(styp,xp,zp,up,z,u,vec_S,mat_V,ξ,λ,μ,μp) 
        L = length(vec_S)
        vec_temp2 = zeros(Float64,L)
        vec_temp2 = z-u
        vec_temp = zeros(Float64,L)
        vec_temp = mat_V'*vec_temp2
    
        ξ1 = zeros(Float64,L)
        ξ2 = zeros(Float64,L)
    
        ξ1 = styp/λ+μp*(zp-up) + μ*vec_temp
        ξ2 = ξ
    
        mat_inv= zeros(Float64,L,L)
        for i in 1:L
            mat_inv[i,i] = 1.0/(vec_S[i]^2/λ+(μ+μp))
        end
        ξ1 = mat_inv*ξ1
        ξ2 = mat_inv*ξ2
    
        ν = (1.0-sum(mat_V*ξ1))/sum(mat_V*ξ2)
        #println(length(ξ1),"\n",length(ξ2),"\n",length(xp))
        #println(ξ2)
        xp = ξ1 + ν*ξ2 
        #println(xp)
        zp = calc_salpha!(1/μp,xp + up)    
        up += xp - zp
    
        z = calc_P!(mat_V*(xp)+u)
        u += mat_V*xp - z
    

    
        return xp,zp,up,z,u
    end

    function calc_P!(x)
        vec_px = zeros(Float64,length(x))
        for i in 1:length(x)
            vec_px[i] = max(x[i],0.0)
        end
    
        return vec_px
    end

    function calc_salpha!(alpha,x)
        L = length(x)
        salpha = zeros(Float64,L)
        for i in 1:L
            if x[i] > alpha
                salpha[i] = x[i] - alpha
            elseif x[i] < - alpha
                salpha[i] = x[i] +  alpha
            else
                salpha[i] = 0.0
            end
     
            
        end
    
        return salpha
    end


    function calc_K!(M,N,omegamax,β,mat_K)
        dτ = β/(M-1)
        dω = 2omegamax/(N-1)
        
        for iome in 1:N
            ω = (iome-1)dω-omegamax 
            for itau in 1:M
                τ = (itau-1)dτ
                td = τ*ω
                if td > 50.0
                    et1 = exp(50.0)
                else
                    et1 = exp(td)
                end
                td =(τ-β)ω
                if td > 50.0
                    et2 = exp(50.0)
                else
                    et2 = exp(td)
                end
                mat_K[itau,iome] = 1.0/(et1+et2)
                #println(mat_K[itau,iome])
                #println(td)
            end
        end
        
    
        return mat_K
    end
end