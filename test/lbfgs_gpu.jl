# Regression for the scalar-indexing error inside the LBFGS operator's
# `lbfgs_multiply`.
#
# `LinearOperators.LBFGSData` previously hard-wired its working buffers
# (`Ax`, the `s`/`y`/`a`/`b` curvature pairs) to `Vector{T}` and
# `Matrix{T}`. When `LBFGSSolver` was built around a `CuVector{T}` NLP,
# the first line of `lbfgs_multiply`, `q .= x`, broadcast a `CuVector`
# into a CPU `Vector{T}` (`LinearOperators/src/lbfgs.jl:128`). The CPU
# broadcast then tried to `getindex` the `CuArray` element-by-element and
# `GPUArraysCore` threw `Scalar indexing is disallowed.`.
#
# The fix parameterises `LBFGSData` (and `LBFGSOperator`) by the working
# vector type `V<:AbstractVector{T}`, threads it through the
# `LBFGSOperator(T, V, n; …)` / `InverseLBFGSOperator(T, V, n; …)`
# constructors, and `JSOSolvers.LBFGSSolver` now passes the NLP's `V`
# through to `InverseLBFGSOperator`. This test exercises the whole
# `lbfgs(nlp)` call on a small `CuVector`-backed unconstrained problem
# under `CUDA.allowscalar(false)`.

using Test
using LinearAlgebra
using CUDA
using JSOSolvers
using NLPModelsTest

if CUDA.functional()
    using LinearOperators: InverseLBFGSOperator
    @testset "InverseLBFGSOperator: push!/mul! with CuVector" begin
        # Smaller, direct regression on the operator itself — pins down both
        # `LBFGSData(T, V, n; …)` (GPU-friendly buffers) *and*
        # `push!(::LBFGSOperator, ::AbstractVector, …)` (was hardcoded to
        # `::Vector{T}` so `push!(H, ::CuVector, ::CuVector)` raised
        # `MethodError`).
        T = Float64
        VT = CuVector{T}
        n = 4
        H = InverseLBFGSOperator(T, VT, n, mem = 3)
        CUDA.allowscalar(false)
        try
            for _ in 1:3
                push!(H, CUDA.randn(T, n), CUDA.randn(T, n) .+ T(0.5))
            end
            d = H * CUDA.randn(T, n)
            @test d isa CuArray
            @test isfinite(sum(d))
        finally
            CUDA.allowscalar(true)
        end
    end

    @testset "lbfgs has no GPU scalar-indexing in `lbfgs_multiply`" begin
        # `BROWNDEN` is the smallest unconstrained NLP in `NLPModelsTest`
        # (n = 4) so the test stays fast. Any unconstrained NLP would do.
        nlp = BROWNDEN(CuArray{Float64, 1, CUDA.DeviceMemory})

        CUDA.allowscalar(false)
        try
            stats = lbfgs(nlp; max_iter = 10)
            # `lbfgs_multiply` must run at least once for the regression to
            # bite, so a non-zero iteration count is the real check.
            @test stats.iter ≥ 1
            # And the standard convergence sanity check.
            @test stats.status in (:first_order, :max_iter)
        finally
            CUDA.allowscalar(true)
        end
    end
else
    @info "CUDA not functional — skipping lbfgs GPU regression"
end
