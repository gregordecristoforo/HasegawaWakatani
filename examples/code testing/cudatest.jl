domain = Domain(256, 256, 128, 128, anti_aliased=true, use_cuda=true)

domain.kx
domain.ky
domain.x
domain.y

domain.SC.Laplacian
domain.SC.invLaplacian
domain.SC.HyperLaplacian
domain.SC.DiffX
domain.SC.DiffY
domain.SC.DiffXX
domain.SC.DiffYY

FT = domain.transform.FT
u = CuArray(ic[:, :, 1])
u_hat = FT * u

diffX(u_hat, domain)           # ✓
diffY(u_hat, domain)           # ✓
diffXX(u_hat, domain)          # ✓
diffYY(u_hat, domain)          # ✓
diffusion(u_hat, domain)       # ✓
hyper_diffusion(u_hat, domain) # ✓
solvePhi(u_hat, domain)        # ✓

spectral_transform!(u_hat, u, domain.transform.FT) # ✓

# QuadraticTerms cache
domain.SC.up                  # ✓
domain.SC.vp                  # ✓
domain.SC.U                   # ✓
domain.SC.V                   # ✓
domain.SC.qtl                 # ✓
domain.SC.qtr                 # ✓
domain.SC.phi                 # ✓
domain.SC.QTPlans.FT          # ✓
domain.SC.QTPlans.iFT         # ✓

# More complex methods
spectral_exp(u_hat, domain)   # ✓
spectral_expm1(u_hat, domain) # ✓
spectral_log(u_hat, domain)   # ✓

# Methods utilizing threading
quadraticTerm(diffXX(u_hat, domain), diffYY(u_hat, domain), domain) # ✓
poissonBracket(u_hat, u_hat, domain)                                # ✓

p = Dict(
    "D" => 1e-2,
    "ν" => 1e-2,
    "g" => 1e-1,
    "σ" => 1e-3,
)

L(u_hat, domain, p, 0)

uu_hat = cat(u_hat, u_hat, dims=3)
N(uu_hat, domain, p, 0)

prob = SpectralODEProblem(L, N, domain, ic, t_span, p=p, dt=2e-3)
cache = get_cache(prob, MSS3())
cache.c

if isnan.(u_hat)
    println("hi")
end


##
#CUDA.@profile 
@time sol = spectral_solve(prob, MSS3(), output, resume=false)
# 173 seconds
# 356 s
# 414 s (All diagnostics and displaying)
# 368 s (No display or probes)
# 390 s (No display, all probes)
# 3078 s (No GPU) 

#CUDA.default_memory = CUDA.UnifiedMemory


total_0 = CUDA.free_memory()
total_0 - CUDA.free_memory()
# 245366784 bytes | 0.25 GB (Domain)
# 4194304 bytes   | 0.004194304 GB (SpectralODEProblem)
# 301989888 bytes | 0.30 GB (Running the program once, most likely allocation for functions)
# Total 0.635 GB (Not shaby)

# --------------------------- Investigating memory leak ------------------------------------
# 33554432 bytes  | 0.03 GB (Running with store_hdf=true)
# 134217728 bytes | 0.134 GB (Running once more)
# 369098752 bytes | 0.369 GB (Out of nowhere)
#-570425344 bytes |-0.570 GB (Freed out of nowhere too)

data_cpu = output.simulation["fields"][:,:,1,end]
heatmap(data_cpu)
heatmap(data_gpu)
heatmap(ic[:,:,1])

CUDA.pool_status()

ic - output.simulation["fields"][:,:,:,1]