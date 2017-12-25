include("ctaux.jl")
module dmft
    export dmft_solver
    import Ctaux

    function dmft_solver(β0,U0,μ0,K0,mqs0,ntime0,mfreq0,norbs0,V0,nthermal0,mkink0,itemax,α)
    
        println( "-----------------------------------------------------------------")
        println( "--Dynamical Mean Field Theory for the Bethe lattice            --")
        println( "--                                                             --")
        println( "--                      Yuki Nagai, Ph.D 12/15/2017(MM/DD/YY)  --")
        println( "-----------------------------------------------------------------")

        println( "-----------------------------------------------------------------")    
        println("DMFT: Initial calculation")        
        τmesh,Gτ,orderdisp,S,Gω,ωmesh = Ctaux.ctaux_solver(β0,U0,μ0,K0,mqs0,ntime0,mfreq0,norbs0,V0,nthermal0,mkink0,false)
        println("Gτ[1] = ",Gτ[1,1]," Gτ[ntime] = ",Gτ[ntime0,1])
    
        #return τmesh,Gτ,Gω,ωmesh
            
        Δ = zeros(Complex{Float64},mfreq0,norbs0,norbs0)
        Δ[:,:,:] = Gω[:,:,:]/4
        Gω_old = zeros(Complex{Float64},mfreq0,norbs0,norbs0)
        Gω_old[:,:,:] = Gω[:,:,:]
        

        println( "-----------------------------------------------------------------")      
        println("DMFT: Loops start!")
        error = 0.0
        for ite in 1:itemax
            τmesh,Gτ,orderdisp,S,Gω,ωmesh = Ctaux.ctaux_solver_general(Δ,β0,U0,μ0,K0,mqs0,ntime0,mfreq0,norbs0,V0,nthermal0,mkink0,false)
            h1 = 0.0
            h2 = 0.0
            for i in 1:norbs0
                h1 += (Gω_old[:,i,i]-Gω[:,i,i])'*(Gω_old[:,i,i]-Gω[:,i,i])
                h2 += Gω_old[:,i,i]'*Gω_old[:,i,i]            
            end
            error = h1/h2
            println(ite,"/",itemax,": ", ite,"-th ","DMFT loop: error = ",error)
            println("Gτ[1] = ",Gτ[1,1]," Gτ[ntime] = ",Gτ[ntime0,1])        
            
            Δ = calc_hyblidization_Bethe!(Δ,Gω,α)
            Gω_old[:,:,:] = Gω[:,:,:]
        end
    
    
        return τmesh,Gτ,Gω,ωmesh,Δ
    
    end

    function calc_hyblidization_Bethe!(Δ,Gω,α)
        Δ = (1-α)*Δ+ α*Gω/4
        return Δ
    end
end