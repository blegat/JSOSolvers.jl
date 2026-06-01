# Regression for the scalar-indexing error inside `apply_inv!` of
# `CompactInverseLBFGS`.
#
# The buggy version called `ldiv!(UpperTriangular(StYk)', inner)`, which
# evaluates to `ldiv!(LowerTriangular(adjoint(StYk)), inner)`. Julia's
# `ldiv!(::AbstractTriangular, ::AbstractVecOrMat)` calls `istriu(A)` to
# pick the upper/lower branch; there's no `istriu(::LowerTriangular)`
# short-circuit, so the generic implementation iterates over the
# off-diagonals — fine on CPU, but on a `CuArray`-backed view it throws
# `Scalar indexing is disallowed.` from `GPUArraysCore`.
#
# The fix calls `LinearAlgebra._ldiv!(inner, A, inner)` directly, skipping
# the `istriu` dispatch. This test exercises the fixed path under
# `CUDA.allowscalar(false)` with a populated `CompactInverseLBFGS`.

using Test
using LinearAlgebra
using CUDA
using JSOSolvers: CompactInverseLBFGS, apply_inv!

if CUDA.functional()
    @testset "CompactInverseLBFGS.apply_inv! has no GPU scalar-indexing" begin
        n   = 8
        mem = 4
        T   = Float64
        VT  = CuVector{T}
        MT  = CuMatrix{T}

        H = CompactInverseLBFGS(T, VT, MT, n, mem)

        # Populate enough curvature pairs to take `apply_inv!` through the
        # full code path (not the early `k == 0` exit). Three pairs is enough
        # to make `StYk` non-trivially adjoint-solved (k = 3 → 3×3 system).
        CUDA.allowscalar() do
            # Build the host pairs first so the curvature is well-defined,
            # then push them in (`push!` itself does no scalar indexing).
            for j in 1:3
                s_cpu = randn(T, n)
                y_cpu = randn(T, n)
                # Ensure `sᵀy > 0` so `γ = sᵀy / yᵀy` is finite & positive.
                y_cpu .+= 0.5 .* s_cpu
                push!(H, CuArray(s_cpu), CuArray(y_cpu))
            end
        end

        g = CUDA.randn(T, n)
        d = CUDA.fill(zero(T), n)

        # The regression check: with scalar indexing disabled globally,
        # `apply_inv!` must not iterate over the triangular off-diagonals.
        CUDA.allowscalar(false)
        try
            apply_inv!(d, H, g)
            @test isfinite(sum(d))
        finally
            CUDA.allowscalar(true)
        end
    end
else
    @info "CUDA not functional — skipping CompactInverseLBFGS GPU regression"
end
