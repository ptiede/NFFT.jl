using NFFT, Reactant

@testset "Reactant NFFT tests" begin
    k = rand(2, 64) .- 0.5
    pl = plan_nfft(NFFTBackend(), Array, k, (64, 64))
    plr = plan_nfft(NFFTBackend(), Reactant.RArray, k, (64, 64))

    x = rand(ComplexF64, 64, 64)
    xr = Reactant.to_rarray(x)

    out = pl*x
    outr = @jit plr*xr
    @test outr ≈ out

    y = pl'*out
    yr = @jit plr'*outr

    @test y ≈ yr
end