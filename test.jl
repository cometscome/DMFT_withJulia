include("./DMFT.jl")
using .dmft

β=30.0 #Inverse temperature
U = 2.0 #Density-Density interaction
μ=U/2 #Chemical potential
K = 1.0
mqs = 300000
ntime = 1024 #number of τs
mfreq = 1024 #number of ωns
norbs = 2 #number of orbitals. norbs = 2 for 1-band model.
V = 1.0 #Strength of the hybridization
nthermal = 1000
mkink = 1024

τmesh,Gτ,Gω,ωmesh,Δ=dmft.dmft_solver(β,U,μ,K,mqs,ntime,mfreq,norbs,V,nthermal,mkink,10,0.3)

println("DMFT loops are done.")
println("The final CTAUX calculation starts.")
mqs = mqs*10
τmesh,Gτ,orderdisp,S,Gω,ωmesh = Ctaux.ctaux_solver_general(Δ,β,U,μ,K,mqs,ntime,mfreq,norbs,V,nthermal,mkink,false)
println("End.")