module Ctaux
    import  Dierckx
    export ctaux_solver,ctaux_solver_general,symmetrize

    function ctaux_solver_general(Δ,β,U,μ,K,mqs,ntime,mfreq,norbs,V,nthermal,mkink,display)
#        global β,U,μ,K,mqs,mfreq,ntime,norbs,V,nthermal,mkink
#        const β=β0
#        const U=U0
#        const μ=μ0
#        const K=K0
#        const mqs = mqs0
#        const mfreq = mfreq0
#        const ntime = ntime0
#        const norbs = norbs0
#        const V = V0
#        const nthermal = nthermal0
#        const mkink = mkink0
        
        
    
        println( "------------------------------------------------------")
        println( "--Continuous-time auxiliary-field Monte Carlo method--")
        println( "--                       for quantum impurity models--")
        println( "--                                                  --")
        println(  "--         See, E. Gull et al., EPL 82, 57003 (2008)--")
        println(  "--             Yuki Nagai, Ph.D 10/24/2017(MM/DD/YY)--")
        println(  "------------------------------------------------------")
        println(  "Parameters")
        println(  "Inverse temperature: β = ",β)
        println(  "Density-Density interaction: U = ",U)
        println(  "Chemical potential: μ = ",μ)
        println(  "Parameter 'K': K = ", K)
        println( "Number of QMC steps: mqs = ",mqs)
    
        #println(currentk)


        println("Initializing...")
        currentk = 0
        nmat,S,Scount,τcon,spincon,indexcon = init_arrays(mkink,norbs,ntime)
        τmesh,ωmesh,Gτ0spl,γ0,expVg= ctaux_init_general(Δ,μ,β,U,K,mkink,ntime,mfreq,norbs)
        #println(currentk)


        println("done.")
#        global γ0
        #println(Ef)
        println("γ0= ",γ0)

        println("Thermalizing...")
        currentk = ctaux_thermal!(nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β,nthermal,norbs,K)            
        println("done.")

        println("QMC Start!")
        @time currentk,orderdisp = ctaux_qmc!(nmat,τcon,spincon,indexcon,S,Scount,currentk,τmesh,Gτ0spl,expVg,β,ntime,norbs,mqs,K,mkink,display)
        println("done!")

        println("Calculating Green's function...")
        Gτ = calc_green(S,τmesh,Gτ0spl,β,ntime,norbs)
        println("done.")
        #Gτ[:,:] =-Gτ[:,:] 
        Gω = fft_τ2ω(Gτ,τmesh,ωmesh,norbs,mfreq,ntime,β)
   
        return τmesh,Gτ,orderdisp,S,Gω,ωmesh
    end

    


    function ctaux_solver(β,U,μ,K,mqs,ntime,mfreq,norbs,V,nthermal,mkink,display)
#        global β,U,μ,K,mqs,mfreq,ntime,norbs,V,nthermal,mkink
#        const β=β0
#        const U=U0
#        const μ=μ0
#        const K=K0
#        const mqs = mqs0
#        const mfreq = mfreq0
#        const ntime = ntime0
#        const norbs = norbs0
#        const V = V0
#        const nthermal = nthermal0
#        const mkink = mkink0
        
        
    
        println( "------------------------------------------------------")
        println( "--Continuous-time auxiliary-field Monte Carlo method--")
        println( "--                       for quantum impurity models--")
        println( "--                                                  --")
        println(  "--         See, E. Gull et al., EPL 82, 57003 (2008)--")
        println(  "--             Yuki Nagai, Ph.D 10/24/2017(MM/DD/YY)--")
        println(  "------------------------------------------------------")
        println(  "Parameters")
        println(  "Inverse temperature: β = ",β)
        println(  "Density-Density interaction: U = ",U)
        println(  "Chemical potential: μ = ",μ)
        println(  "Parameter 'K': K = ", K)
        println( "Number of QMC steps: mqs = ",mqs)
    
        #println(currentk)


        println("Initializing...")
        currentk = 0
        nmat,S,Scount,τcon,spincon,indexcon = init_arrays(mkink,norbs,ntime)
        τmesh,ωmesh,Gτ0spl,γ0,expVg = ctaux_init(μ,β,U,K,V,mkink,ntime,mfreq,norbs)
        #println(currentk)


        println("done.")
#        global γ0
        #println(Ef)
        println("γ= ",γ0)

        println("Thermalizing...")
        currentk = ctaux_thermal!(nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β,nthermal,norbs,K)        
        println("done.")

        println("QMC Start!")
        @time currentk,orderdisp = ctaux_qmc!(nmat,τcon,spincon,indexcon,S,Scount,currentk,τmesh,Gτ0spl,expVg,β,ntime,norbs,mqs,K,mkink,display)
        println("done!")

        println("Calculating Green's function...")
        Gτ = calc_green(S,τmesh,Gτ0spl,β,ntime,norbs)
        println("done.")
        #Gτ[:,:] =-Gτ[:,:] 
        Gω = fft_τ2ω(Gτ,τmesh,ωmesh,norbs,mfreq,ntime,β)
   
        return τmesh,Gτ,orderdisp,S,Gω,ωmesh
    end

    function ctaux_init(μ,β,U,K,V,mkink,ntime,mfreq,norbs)
#        global orderdisp,mkink
#        orderdisp = zeros(Int64,mkink)
    
        #global Dr,Dc,Dd,nmat
        global debug
        debug = false
        #=
        global orderdisp,detold,debug,averagesign,accept_insert,accept_remove,number_insert,number_remove,averagek

        detold = ones(Float64,2)
        
        averagesign = 0
        accept_insert = 0
        accept_remove = 0
        number_insert = 0
        number_remove = 0
        averagek = 0
        =#



        srand(1234)
    
#        global μ,β,U,K
#        global Ef
        Ef = -μ
#        global γ0,expVg
        
        γ0=acosh(1.0+(β*U)/(2.0*K))
        expVg = (exp(γ0),exp(-γ0))
        #println("exp",expVg,"γ",γ0)
        #global τmesh
        τmesh = calc_linear_mesh(0.0,β,ntime) 
        #global ωmesh
        ωmesh = calc_linear_mesh(π/β,(π/β)*(2mfreq-1.0),mfreq)
        #global Δ
        Δ = zeros(Complex{Float64},mfreq,norbs,norbs)
        for i in 1:mfreq
            for j in 1:norbs
                Δ[i,j,j]=1.0
            end
            #Hybridization function for the Bethe lattice
            z = ωmesh[i]*im
            Δ[i,:,:]=Δ[i,:,:]*(z-im*sqrt(1-z^2))/2
        end
        Δ = V*V*Δ
        #ln(Δ[:,1,1])
        
        Gf0 = zeros(Complex{Float64},mfreq,norbs,norbs) #Non-perturbative Green's function in the Matsubara space
        for i in 1:mfreq
            for j in 1:norbs
                Gf0[i,j,j] = 1/(im*ωmesh[i] - Ef - Δ[i,j,j] - U/2)
            end
        end
        
        Gτ0 = zeros(Float64,ntime,norbs,norbs) #Non-perturbative Green's function in the imaginary-time space
        Gτ0 = fft_ω2τ(Gf0,ωmesh,norbs,mfreq,ntime,β)
        #testGω = fft_τ2ω(Gτ0,τmesh,ωmesh)
        #for i in 1:mfreq
        #    println(i,", ",Gf0[i,1,1],", ",testGω[i,1,1])
        #end
        
        #ln(typeof(Gτ0))
        global Gτ0spl
        Gτ0spl0 = []
        Gf0 = - Gf0 
        Gτ0 = - Gτ0
        #println(Gτ0)
        for i in 1:norbs
#            testv = Gτ0[:,i,i]
            spl = Dierckx.Spline1D(τmesh, Gτ0[:,i,i])
            #println(spl(3.0))
            push!(Gτ0spl0,spl)
        end 
        Gτ0spl=Gτ0spl0

        return τmesh,ωmesh,Gτ0spl,γ0,expVg
        
    end

    function ctaux_init_general(Δ,μ,β,U,K,mkink,ntime,mfreq,norbs)
        #global orderdisp,mkink
#        orderdisp = zeros(Int64,mkink)
    
        #global Dr,Dc,Dd,nmat
        global debug
        debug = false

        srand(1234)
    
#        global μ,β,U,K
#        global Ef
        Ef = -μ
#        global γ0,expVg
        
        γ0=acosh(1.0+(β*U)/(2.0*K))
        expVg = (exp(γ0),exp(-γ0))
        τmesh = calc_linear_mesh(0.0,β,ntime) 
        ωmesh = calc_linear_mesh(π/β,(π/β)*(2mfreq-1.0),mfreq)
        
        Gf0 = zeros(Complex{Float64},mfreq,norbs,norbs) #Non-perturbative Green's function in the Matsubara space
        for i in 1:mfreq
            for j in 1:norbs
                Gf0[i,j,j] = 1/(im*ωmesh[i] - Ef - Δ[i,j,j] - U/2)
            end
        end
        
        Gτ0 = zeros(Float64,ntime,norbs,norbs) #Non-perturbative Green's function in the imaginary-time space
        Gτ0 = fft_ω2τ(Gf0,ωmesh,norbs,mfreq,ntime,β)
        #testv = zeros(Float64,ntime)
        
        #ln(typeof(Gτ0))
#        global Gτ0spl
        Gτ0spl0 = []
        Gf0 = - Gf0 
        Gτ0 = - Gτ0
        #println(Gτ0)
        for i in 1:norbs
#            testv = Gτ0[:,i,i]
            spl = Dierckx.Spline1D(τmesh, Gτ0[:,i,i])
            #println(spl(3.0))
            push!(Gτ0spl0,spl)
        end 
        #const Gτ0spl=Gτ0spl0
        Gτ0spl=Gτ0spl0

        return τmesh,ωmesh,Gτ0spl,γ0,expVg
        
    end

    function symmetrize(Gω,τmesh,ωmesh,norbs,mfreq,ntime,β)
        Gτnew = zeros(Float64,ntime,norbs)  
        Gωnew = zeros(Complex{Float64},mfreq,norbs,norbs) 
        Gωnew[:,:,:] = im*imag(Gω[:,:,:])
        Gτnew = -fft_ω2τ(Gωnew,ωmesh,norbs,mfreq,ntime,β)
        return Gτnew
    end

    function ctaux_thermal!(nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β,nthermal,norbs,K)        
        averagesign = 0.0
        for i in 1:nthermal
            r = rand()
            
            if r > 0.5
                currentk,pass,rsign = ctaux_insert!(nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β,norbs,K)
                averagesign += rsign
            else
                currentk,pass,rsign = ctaux_remove!(nmat,τcon,spincon,indexcon,currentk,K)
                averagesign += rsign
            end
            

        end
        println("Average sign = ",averagesign/nthermal)


        #=
        global mkink
        for i in 1:mkink
            #println(i-1,"\t",orderdisp[i])
        end
        =#
        return currentk
    end




    function ctaux_qmc!(nmat,τcon,spincon,indexcon,S,Scount,currentk,τmesh,Gτ0spl,expVg,β,ntime,norbs,mqs,K,mkink,display)
        #global averagesign
        number_insert = 0
        number_remove = 0
        accept_insert = 0
        accept_remove = 0       
        averagesign = 0
        averagek = 0
        orderdisp = zeros(Int64,mkink)
        for i in 1:mqs
            r = rand()  
            
            averagek += currentk
            if r > 0.5
                number_insert += 1
                currentk,pass,rsign = ctaux_insert!(nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β,norbs,K)
                averagesign += rsign

                if pass 
                    accept_insert += 1
                    orderdisp[currentk+1] += 1
                    

                end

            else
                number_remove += 1
                currentk,pass,rsign = ctaux_remove!(nmat,τcon,spincon,indexcon,currentk,K)
                averagesign += rsign
                #println(pass)
                if pass 
                    accept_remove += 1
                    orderdisp[currentk+1] += 1
                end

            end
            

            if i % (mqs/10) == 0 && display == true
                println("---------------------------------------------")
                println("Statistics")
                println("Number of QMC steps: ",i," of ",mqs)
                println("Average order k: ",averagek/i)
                println("Average sign :", averagesign/i)
                println("Insertion updates")
                println("Total","\t","accepted","\t","rate")
                println(number_insert,"\t",accept_insert,"\t",accept_insert/number_insert)
                println("Removal updates")
                println("Total","\t","accepted","\t","rate")
                println(number_remove,"\t",accept_remove,"\t",accept_remove/number_remove)
            end

            calc_S!(nmat,τcon,spincon,indexcon,S,Scount,currentk,τmesh,Gτ0spl,expVg,β,ntime,norbs)

        end
        

        S[:,:] = S[:,:]/mqs
        Scount[:]=ntime*Scount[:]/sum(Scount)
#        global ntime
        for i in 1:ntime
            S[i,:] = S[i,:]/Scount[i]
        end

        return currentk,orderdisp

    end

    


    function ctaux_insert!(nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β,norbs,K)
#        println("-----------insert----------------------")
        pass = false
        vertex = try_insert_vertex(τcon,spincon,indexcon,currentk,β) #trial vertex (τ,spin,index)

    
        det_ratioup = 0.0
        det_ratiodown = 0.0

        #global norbs

        Dr = zeros(Float64,currentk,norbs)
        Dc = zeros(Float64,currentk,norbs)


        det_ratioup = calc_insert!(vertex,1,Dr,Dc,nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β)
        det_ratiodown = calc_insert!(vertex,2,Dr,Dc,nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β)

        #=
        global debug
        if debug 
            detup = calc_insert_det(vertex,1)
            detdown = calc_insert_det(vertex,2)
            global detold
        
            println("r:",det_ratioup,"\t",detup/detold[1])
            println("r:",det_ratiodown,"\t",detdown/detold[2])
        end
        =#

        ratio = det_ratioup*det_ratiodown*K/(currentk+1)
        rsign = sign(ratio)
#        global averagesign

        
#        println("ratio insert:",ratio,"\t",currentk)
        r = rand()

        if min(1.0,abs(ratio)) > r

            vertex_insert!(vertex,τcon,spincon,indexcon,currentk)

            currentk += 1

            pass = true

            #=
            if debug
                detold[1] = detup
                detold[2] = detdown
            end
            =#
           
            construct_insert_nmat!(vertex,1,det_ratioup,Dr,Dc,nmat,indexcon,currentk)
            construct_insert_nmat!(vertex,2,det_ratiodown,Dr,Dc,nmat,indexcon,currentk)

        end    
        return currentk,pass,rsign
    end

    function ctaux_remove!(nmat,τcon,spincon,indexcon,currentk,K)
        pass = false
        if currentk ==0
            rsign = 1.0
            return currentk,pass,rsign
        end

        vertex = try_remove_vertex(τcon,spincon,indexcon,currentk) #trial vertex (τ,spin,index) 

        det_ratioup = calc_remove(vertex,1,nmat,indexcon)
        det_ratiodown = calc_remove(vertex,2,nmat,indexcon)    

        #=
        global debug
        if debug
            detup = calc_remove_det(vertex,1)
            detdown = calc_remove_det(vertex,2)
            global detold
            println("r:",det_ratioup,"\t",detup/detold[1])
            println("r:",det_ratiodown,"\t",detdown/detold[2])
        end
        =#

#        global K
        ratio = (currentk/K)*det_ratioup*det_ratiodown
        rsign = sign(ratio)

        
        
        r = rand()
        #println("r: ",r)
        if min(1.0,abs(ratio)) > r
            as = indexcon[vertex[3]]
            vertex_remove!(vertex,τcon,spincon,indexcon,currentk)
            pass = true

            #=
            if debug
                detold[1] = detup
                detold[2] = detdown
            end
            =#
            currentk += -1

            construct_remove_nmat!(as,1,det_ratioup,nmat,currentk)
            
            construct_remove_nmat!(as,2,det_ratiodown,nmat,currentk)
            

        end
        return currentk,pass,rsign
    end



    function calc_insert!(vertex,rspin,Dr,Dc,nmat,τcon,spincon,indexcon,currentk,Gτ0spl,expVg,β) #fast update
        ssign = (-1)^(rspin-1)
#        global γ,expVg
        k = currentk
        
        
        Gτ0 = Gτ0spl[rspin](1e-8)
        #expV = exp(γ*ssign*vertex[2])
        ev =  ifelse(ssign*vertex[2]==1,expVg[1],expVg[2])
        #println(ev,"\t",expV,"\t",ssign*vertex[2],"\t",γ)
        Dd = ev - Gτ0*(ev-1)  

        if k ==0
            #println("k = 0!")
            det_ratio = Dd
            return det_ratio
        end


        τj = vertex[1]
        sj = vertex[2]
        for i in 1:k
            τi = τcon[indexcon[i]]
            Gτ0 = calc_Gτ(τi,τj,rspin,β,Gτ0spl)
            ev =  ifelse(ssign*sj==1,expVg[1],expVg[2])
            Dc[indexcon[i],rspin] = - Gτ0*(ev-1)
        end


        τi = vertex[1]
        #println("τs: ",τi)
        for j in 1:currentk
            jj = indexcon[j]
            τj = τcon[jj]
            #println("τj:",τj,"\t j",indexcon[j])
            sj = spincon[jj]
            Gτ0 = calc_Gτ(τi,τj,rspin,β,Gτ0spl)
            ev =  ifelse(ssign*sj==1,expVg[1],expVg[2])
            Dr[indexcon[j],rspin] = - Gτ0*(ev-1)
        end

        #=
        λ = Dd
        for i in 1:currentk
            for j in 1:currentk
                λ+= - Dr[i,rspin]*nmat[i,j,rspin]*Dc[j,rspin]
            end
        end
        =#

        λ = Dd -dot(Dr[1:currentk,rspin],nmat[1:currentk,1:currentk,rspin]*Dc[1:currentk,rspin])
        #det_ratio = λ
        det_ratio = λ[1]
    
        return det_ratio
    end

    function calc_remove(vertex,rspin,nmat,indexcon)
#        global nmat,indexcon
        ind = indexcon[vertex[3]]
        det_ratio = nmat[ind,ind,rspin] 
        return det_ratio
    end

    function calc_Gτ(τi,τj,sigma,β,Gτ0spl)
        dτ = τi - τj
        if dτ < 0
            dτ += β
            Gτ0 = -Gτ0spl[sigma](dτ)
        elseif dτ == 0.0
            Gτ0 = Gτ0spl[sigma](1e-8)
        else
            Gτ0 = Gτ0spl[sigma](dτ)
        end
        return Gτ0
    end



    function try_insert_vertex(τcon,spincon,indexcon,currentk,β)
        #global currentk,τcon,indexcon
        #global currentk
#        global β
        τ=rand()*β
        τcheck = 1
        ε = 1e-12
        if currentk != 0
            while τcheck > 0
                τ=rand()*β
                τcheck = 0
                for i in 1:currentk
                    τcheck += ifelse(abs(τ-τcon[indexcon[i]]) < ε,1,0)
                end
            end
        end

    
        spin = (-1)^(rand(1:2)-1)
        index = 1
    
    

        if(currentk > 0)
            if τ < τcon[indexcon[1]]
                index = 1
            elseif τ > τcon[indexcon[currentk]]
                index = currentk + 1
            else
                i = 1
                for j in 1:currentk
                    i += ifelse(τcon[indexcon[j]] < τ,1,0)
                    #=
                    if τcon[indexcon[j]] < τ
                       
                        i += 1
                    end
                    =#
                end
                index = i
            end
        end

        vertex = (τ,spin,index)

        return vertex
    end

    function try_remove_vertex(τcon,spincon,indexcon,currentk)
#        global currentk,indexcon
#        global currentk

        index = rand(1:currentk)
        τ = τcon[indexcon[index]]
        spin = spincon[indexcon[index]]
        vertex = (τ,spin,index)


        return vertex
    end




    function vertex_insert!(vertex,τcon,spincon,indexcon,currentk)

         #= old methods like Fortran language
        index = currentk + 1
        for i in currentk:-1:vertex[3]
            indexcon[i+1] = indexcon[i]
        end

        indexcon[vertex[3]] = index
        τcon[index] = vertex[1]
        spincon[index] = vertex[2]
        =#

        
        if currentk == 0
            index = 1
 
            insert!(indexcon,1,index)
            insert!(τcon,1,vertex[1])
            insert!(spincon,1,vertex[2])

            return
        end
        index = currentk+1 
        insert!(indexcon,vertex[3],index)
        insert!(τcon,index,vertex[1])
        insert!(spincon,index,vertex[2])
        

    end

    function vertex_remove!(vertex,τcon,spincon,indexcon,currentk)
#        global currentk

        #= old method like Fortran language        
        index = indexcon[vertex[3]]
        for i in vertex[3]:currentk-1
            indexcon[i] = indexcon[i+1]
        end
        indexcon[currentk] = 0
        for i in 1:currentk
            if indexcon[i] > index
                indexcon[i] = indexcon[i]-1
            end
        end

        if index != currentk
            for i in index:currentk-1
                τcon[i] = τcon[i+1]
                spincon[i] = spincon[i+1]
            end
        end        
        =#

             
        is = indexcon[vertex[3]]
        deleteat!(τcon,is)
        deleteat!(spincon,is)
        deleteat!(indexcon,vertex[3])

        for i in 1:currentk-1
            indexcon[i] += ifelse(indexcon[i] >= is,-1,0)
            #=
            if indexcon[i] >= is
                indexcon[i] += -1
            end
            =#
        end
        
    end



    function construct_insert_nmat!(vertex,rspin,ratio,Dr,Dc,nmat,indexcon,currentk)
        k = currentk
        if k == 1
            nmat[1,1,rspin] = 1/ratio
            return
        end
        λ=ratio
        R = zeros(Float64,k-1)
        R = - nmat[1:k-1,1:k-1,rspin]'*Dr[1:k-1,rspin]/λ

        L = zeros(Float64,k-1)
        L = - nmat[1:k-1,1:k-1,rspin]*Dc[1:k-1,rspin]/λ

        ntemp = zeros(Float64,k,k)
        nmatt = zeros(Float64,k-1,k-1)
        nmatt[1:k-1,1:k-1]=nmat[1:k-1,1:k-1,rspin]

        for i in 1:k-1
            for j in 1:k-1
                nmatt[i,j] += λ*L[i]*R[j]
            end
        end
        as = indexcon[vertex[3]]        
        ntemp[as,as] = 1/λ
        #println("as ",as,"\t",k)
        ntemp[1:k-1,1:k-1] = nmatt[1:k-1,1:k-1]
        ntemp[k,1:k-1] = R[1:k-1]
        ntemp[1:k-1,k] = L[1:k-1]        

        #=
        if as == 1
            ntemp[2:k,2:k] = nmatt[1:k-1,1:k-1]
            ntemp[1,2:k] = R[1:k-1]
            ntemp[2:k,1] = L[1:k-1]
        elseif as == k
            ntemp[1:k-1,1:k-1] = nmatt[1:k-1,1:k-1]
            ntemp[k,1:k-1] = R[1:k-1]
            ntemp[1:k-1,k] = L[1:k-1]

        else
            ntemp[1:as-1,1:as-1] = nmatt[1:as-1,1:as-1]
            ntemp[1:as-1,as] = L[1:as-1]
            ntemp[as+1:k,as] = L[as:k-1]
            ntemp[as,1:as-1] = R[1:as-1]
            ntemp[as,as+1:k] = R[as:k-1]
        end
        =#

        nmat[1:k,1:k,rspin] = ntemp[1:k,1:k]
    
    end

    function construct_remove_nmat!(as,rspin,det_ratio,nmat,currentk)
#        global currentk
        k = currentk
        if currentk == 0
            return
        end
        

        λ = det_ratio
        ntemp = zeros(Float64,k,k)
        for j in 1:k
            #=
            if j -as >= 0
                jtheta = 1
            else 
                jtheta = 0
            end
            jj = j + jtheta
            =#
            jj = j + ifelse(j - as>=0,1,0)

            for i in 1:k
                #=
                if i-as >= 0
                    itheta = 1
                else
                    itheta = 0
                end
                ii = i + itheta
                =#
                ii = i + ifelse(i - as >= 0,1,0)
                ntemp[i,j] = nmat[ii,jj,rspin] -nmat[ii,as,rspin]*nmat[as,jj,rspin]/λ
            end
        end
        nmat[1:k,1:k,rspin] = ntemp[1:k,1:k]

    end



    function init_arrays(mkink,norbs,ntime)
        nmat = zeros(Float64,mkink,mkink,norbs)
        #global S,Scount
        S = zeros(Float64,ntime,norbs)
        Scount = zeros(Float64,ntime,norbs)
   
        #=
        τcon = zeros(Float64,mkink)    
        spincon = zeros(Int64,mkink)
        indexcon = zeros(Int64,mkink) #address of configurations
        =#

        
        τcon = Float64[]
        spincon = Int64[]
        indexcon = Int64[]
        

        return nmat,S,Scount,τcon,spincon,indexcon
    end

    function calc_green(S,τmesh,Gτ0spl,β,ntime,norbs)
#        global β,ntime,norbs
        δτ = β/(ntime-1)
        Gτ = zeros(Float64,ntime,norbs)
        gsum = zeros(Float64,norbs)
        g0 = zeros(Float64,norbs)


        for i in 1:ntime
            τ = τmesh[i]
            if i==ntime
                τ += -1e8
            elseif i==1
                τ += 1e-8
            end

            for sigma in 1:norbs
                Gτ[i,sigma] = Gτ0spl[sigma](τ)
            end
            gsum[:] =0.0
            for j in 1:i
                τt = τmesh[j]
                dτ = τ-τt
                if i == j
                    dτ = 1e-8
                end

                for sigma in 1:2
                    g0[sigma] = Gτ0spl[sigma](dτ)
                    gsum[sigma] += g0[sigma]*S[j,sigma]*δτ*ifelse(j == 1 || j ==i,0.5,1.0)
                    
                    #=
                    if j == 1 || j ==i
                        gsum[sigma] += g0[sigma]*S[j,sigma]*δτ/2
                    else
                        gsum[sigma] += g0[sigma]*S[j,sigma]*δτ
                    end
                    =#


                end
            
            end    
            for j in i:ntime
                τt = τmesh[j]
                dτ = τ-τt
                if i == j
                    dτ = -1e-8
                end
                for sigma in 1:2
                    g0[sigma] = -Gτ0spl[sigma](dτ+β)
                    gsum[sigma] += g0[sigma]*S[j,sigma]*δτ*ifelse(j == ntime || j ==i,0.5,1.0)
                    #=
                    if j == ntime || j ==i
                        gsum[sigma] += g0[sigma]*S[j,sigma]*δτ/2
                    else
                        gsum[sigma] += g0[sigma]*S[j,sigma]*δτ
                    end
                    =#
                    

                end


            end
            for sigma in 1:norbs
                Gτ[i,sigma] += gsum[sigma]
            end
        end


        return Gτ
    end



    function calc_S!(nmat,τcon,spincon,indexcon,S,Scount,currentk,τmesh,Gτ0spl,expVg,β,ntime,norbs)
        Mklm = zeros(Float64,currentk,currentk,norbs)
        ev = zeros(Float64,norbs)
        g0l = zeros(Float64,currentk,norbs)
        #global γ

        for k in 1:currentk
            kk = indexcon[k]
            τk = τcon[kk]
            sk = spincon[kk]
            for sigma in 1:norbs
                ssign = (-1)^(sigma-1)
                ev[sigma] =  ifelse(ssign*sk==1,expVg[1],expVg[2])
                #ev[sigma] = exp(γ*ssign*sk)
            end
            for l in 1:currentk
                ll = indexcon[l]
                τl = τcon[ll]
                for sigma in 1:norbs
                    Mklm[kk,ll,sigma] = (ev[sigma]-1)*nmat[kk,ll,sigma]
                end
            end
        end

        for l in 1:currentk
            ll = indexcon[l]
            τl = τcon[ll]
            for sigma in 1:norbs
                g0l[ll,sigma] = Gτ0spl[sigma](τl)
            end
        end

#        global ntime,β
        Mkl = zeros(Float64,norbs)
        ξ = β/ntime
        id = 3
        for k in 1:currentk
            kk = indexcon[k]
            τk = τcon[kk]
            iτ = ceil(Int64,ntime*τk/β)
            Mkl[:] = 0.0#zeros(Float64,norbs)
            for l in 1:currentk
                ll = indexcon[l]
                for sigma in 1:norbs                
                    Mkl[sigma] += Mklm[kk,ll,sigma]*g0l[ll,sigma]
                end
            end
            
            for j in -id:id
                jj = iτ+j
                if 1 <= jj <= ntime
                    τd = τmesh[jj]
                    f = gauss(ξ,τd,τk)
                    for sigma in 1:norbs
                        S[jj,sigma] += f*Mkl[sigma]
                    end
                    Scount[jj] += f
                end
            end

        end
        

    end

    function gauss(ξ,x,x0)
        f = (1/(sqrt(2π)*ξ))*exp(-(x-x0)^2/(2*ξ^2))
        return f
    end

    function fft_τ2ω(fτ,τmesh,ωmesh,norbs,mfreq,ntime,β)
#        global norbs,mfreq,ntime,β
        fω = zeros(Complex{Float64},mfreq,norbs,norbs)
        for i in 1:norbs
            fω[:,i,i] = ft_forward(mfreq,ntime,β,fτ[:,i],τmesh,ωmesh)
        end
        return fω
    end

    function ft_forward(mfreq,ntime,β,vτ,τmesh,ωmesh)
        vω = zeros(Complex{Float64},mfreq)
        for i in 1:mfreq
            ωn = ωmesh[i]
            for j in 1:ntime-1
                fa = vτ[j+1]
                fb = vτ[j]
                a = τmesh[j+1]
                b = τmesh[j]
                vω[i] += exp(im*a*ωn)*(-fa+fb+im*(a-b)*fa*ωn)/((a-b)*ωn^2)
                vω[i] += exp(im*b*ωn)*(fa-fb-im*(a-b)*fb*ωn)/((a-b)*ωn^2)
            end
        end
        return vω
    end



    function fft_ω2τ(fω,ωmesh,norbs,mfreq,ntime,β)
#        global norbs,mfreq,ntime,β
        fτ = zeros(Float64,ntime,norbs,norbs)
        for i in 1:norbs
            for j in 1:norbs
                fτ[:,i,j] = fft_backward(mfreq,ntime,β,fω[:,i,j],ωmesh)
            end
        end
                
        for i in 1:norbs
            for j in 1:ntime
                if fτ[j,i,i] > 0.0
                    fτ[j,i,i] = -1e-6 #to avoid positive values
                end
            end
        end
        #println(typeof(fτ))
        return fτ
    end

    function fft_backward(mfreq,ntime,β,vω,ωmesh)
        tail = calc_tails(mfreq,vω,ωmesh)
        gk = zeros(Complex{Float64},2mfreq)
    
        for j in 1:mfreq
            gk[2j] = vω[j] - tail/(im*ωmesh[j])
        end
               
        fft!(gk)
  
        
        gτ = zeros(Float64,ntime)
        gτ[1:ntime]= real(gk[1:ntime])*(2/β)-tail/2
     
        a = real(vω[mfreq])*ωmesh[mfreq]/π
        gτ[1] += a
        gτ[ntime] += -a

        return gτ
       
    
    end

    function calc_tails(mfreq,vω,ωmesh) #to calculate a tail
        ntail = 128
    
        Sn = 0.0
        Sx = 0.0
        Sy = 0.0
    
        Sxx = 0.0
        Sxy = 0.0
    
        for j in mfreq-ntail:mfreq
            ωn = ωmesh[j]
            Sn += 1
            Sx += 1/ωn^2
            Sy += imag(vω[j])*ωn
            Sxx += 1/ωn^4
            Sxy += imag(vω[j])*ωn/ωn^2
        end
            rtail = (Sx*Sxy-Sxx*Sy)/(Sn*Sxx - Sx*Sx)
    
        return rtail
    end

    function calc_linear_mesh(xmin,xmax,n)
        x = zeros(Float64,n)
        for i in 1:n
            x[i] = (xmax-xmin)*(i-1)/(n-1)+xmin
        end        
        return x
    end


#=
----------for debuging.
-----------------------------------------------------------
-----------------------------------------------------------
-----------------------------------------------------------
=#

    function calc_insert_det(vertex,rspin,Gτ0spl,γ) #for debug. slow update
        global τcon,spincon,indexcon,currentk,β
        ssign = (-1)^(rspin-1)
        detn = 0.0

#        global γ

        if currentk ==0
            Gτ0 = Gτ0spl[rspin](1e-8)
            expV = exp(γ*ssign*vertex[2])
            Dd = expV - Gτ0*(expV-1)  

            detn = Dd
            return detn
        end

        indexconold = copy(indexcon)
        τconold = copy(τcon)
        spinconold = copy(spincon)


        vertex_insert(vertex)
        #println("2",τcon)

        mat_D = calc_det_D(rspin,currentk+1,Gτ0spl)

        indexcon = copy(indexconold)
        τcon = copy(τconold)
        spincon = copy(spinconold)



        #println("3",τcon)
        #println("4",τconold)
        

        global detoldd
        detn = det(mat_D)


        return detn


    end

    function calc_remove_det(vertex,rspin,Gτ0spl,γ) #for debug. slow update
        global τcon,spincon,indexcon,currentk,β

        detn = 0.0

#        global γ

        if currentk ==0
            detn = 0.0
            return detn
        end

        indexconold = copy(indexcon)
        τconold = copy(τcon)
        spinconold = copy(spincon)



        vertex_remove(vertex)


        mat_D = calc_det_D(rspin,currentk-1,Gτ0spl)

        indexcon = copy(indexconold)
        τcon = copy(τconold)
        spincon = copy(spinconold)

        detn = det(mat_D)
        return detn

    end


    function calc_det_D(rspin,k,Gτ0spl)
        ssign = (-1.0)^(rspin-1)
      
        mat_D = zeros(Float64,k,k)

        expV = zeros(Float64,k)
        for i in 1:k
            expV[indexcon[i]] = exp(γ*ssign*spincon[indexcon[i]])
        end

        mat_G = zeros(Float64,k,k)
        for j in 1:k
            jj = indexcon[j]
            τj = τcon[jj]
            for i in 1:k
                ii = indexcon[i]
                τi = τcon[ii]
                mat_G[ii,jj] = calc_Gτ(τi,τj,rspin,β)
            end
        end

        for j in 1:k
            for i in 1:k
                if i == j
                    mat_D[i,j] = mat_D[i,j] + expV[i]
                end
                mat_D[i,j] = mat_D[i,j] - mat_G[i,j]*(expV[j]-1)
            end
        end
        return mat_D

    end




end
