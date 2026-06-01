export compact_lbfgs, CompactLBFGSSolver, CompactLBFGSParameterSet, CompactInverseLBFGS

# Default algorithm parameter values
const COMPACT_LBFGS_mem = DefaultParameter(5)
const COMPACT_LBFGS_τ₁ = DefaultParameter(nlp -> eltype(nlp.meta.x0)(0.9999), "T(0.9999)")
const COMPACT_LBFGS_bk_max = DefaultParameter(25)

"""
    CompactLBFGSParameterSet{T} <: AbstractParameterSet

Parameter set for [`compact_lbfgs`](@ref).
"""
struct CompactLBFGSParameterSet{T} <: AbstractParameterSet
  mem::Parameter{Int, IntegerRange{Int}}
  τ₁::Parameter{T, RealInterval{T}}
  bk_max::Parameter{Int, IntegerRange{Int}}
end

function CompactLBFGSParameterSet(
  nlp::AbstractNLPModel{T};
  mem::Int = get(COMPACT_LBFGS_mem, nlp),
  τ₁::T = get(COMPACT_LBFGS_τ₁, nlp),
  bk_max::Int = get(COMPACT_LBFGS_bk_max, nlp),
) where {T}
  CompactLBFGSParameterSet(
    Parameter(mem, IntegerRange(Int(1), Int(100))),
    Parameter(τ₁, RealInterval(T(0), T(1), lower_open = true)),
    Parameter(bk_max, IntegerRange(Int(1), Int(100))),
  )
end

"""
    CompactInverseLBFGS{T, V, M}

Limited-memory BFGS approximation of the **inverse Hessian**, stored in the
compact form of Byrd, Nocedal & Schnabel (1994):

    Hₖ = γₖ I + [Sₖ  γₖ Yₖ] M [Sₖᵀ; γₖ Yₖᵀ]

where `M` is the 2k × 2k middle matrix built from `Rₖ = triu(Sₖᵀ Yₖ)`,
`Dₖ = diag(R)` and `YₖᵀYₖ`. Compared to the standard two-loop recursion, every
product `H g` is reduced to two `n × k` gemv calls plus a couple of `k × k`
triangular solves, which is what the GPU prefers (one large kernel beats
2k tiny `dot`/`axpy` launches).

Type parameters:
- `T`: element type
- `V`: working vector type (e.g. `Vector{T}`, `CuVector{T}`)
- `M`: working matrix type (e.g. `Matrix{T}`, `CuMatrix{T}`)
"""
mutable struct CompactInverseLBFGS{T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
  mem::Int
  cur::Int
  n::Int
  γ::T
  S::M           # n × mem
  Y::M           # n × mem
  StY::M         # mem × mem (its upper triangle is Rₖ)
  YtY::M         # mem × mem (symmetric)
  D::V           # mem (diagonal of Rₖ)
  scratchS::M    # n × mem buffer for in-place shifts
  scratchD::V    # mem buffer for in-place shifts of D
  a1::V
  a2::V
  u1::V
  inner::V
  nprod::Int
end

"""
    CompactInverseLBFGS(::Type{T}, ::Type{V}, ::Type{M}, n::Int, mem::Int)

Allocate all working buffers on whichever device `V`/`M` lives on. The matrix
type `M` is normally derived automatically from `V`; if `V` is `Vector{T}` then
`M` is `Matrix{T}`, if `V` is `CuArray{T, 1, …}` then `M` is the corresponding
2-D `CuArray`.
"""
function CompactInverseLBFGS(::Type{T}, ::Type{V}, ::Type{M}, n::Int, mem::Int) where {T, V, M}
  S        = fill!(M(undef, n, mem), zero(T))
  Y        = fill!(M(undef, n, mem), zero(T))
  StY      = fill!(M(undef, mem, mem), zero(T))
  YtY      = fill!(M(undef, mem, mem), zero(T))
  D        = fill!(V(undef, mem), zero(T))
  scratchS = M(undef, n, mem)
  scratchD = V(undef, mem)
  a1       = V(undef, mem)
  a2       = V(undef, mem)
  u1       = V(undef, mem)
  inner    = V(undef, mem)
  return CompactInverseLBFGS{T, V, M}(
    mem, 0, n, one(T),
    S, Y, StY, YtY, D, scratchS, scratchD,
    a1, a2, u1, inner, 0,
  )
end

function LinearOperators.reset!(H::CompactInverseLBFGS{T}) where {T}
  H.cur = 0
  H.γ = one(T)
  H.nprod = 0
  return H
end

"""
    push!(H::CompactInverseLBFGS, s, y)

Append the curvature pair `(s, y)` to the memory. When the buffer is full, the
oldest pair is dropped via an in-place shift through `H.scratchS` / `H.scratchD`
(no GPU scalar indexing).
"""
function Base.push!(H::CompactInverseLBFGS{T, V, M}, s::AbstractVector{T}, y::AbstractVector{T}) where {T, V, M}
  sTy = dot(s, y)
  yTy = dot(y, y)

  if H.cur < H.mem
    k = H.cur + 1
    @views H.S[:, k] .= s
    @views H.Y[:, k] .= y
    @views H.D[k:k] .= sTy
    H.cur = k
  else
    m = H.mem
    # Shift S, Y left by one column through scratchS (avoids the read/write
    # overlap that would race on GPU).
    @views copyto!(H.scratchS[:, 1:m-1], H.S[:, 2:m])
    @views copyto!(H.S[:, 1:m-1], H.scratchS[:, 1:m-1])
    @views copyto!(H.scratchS[:, 1:m-1], H.Y[:, 2:m])
    @views copyto!(H.Y[:, 1:m-1], H.scratchS[:, 1:m-1])
    @views H.S[:, m] .= s
    @views H.Y[:, m] .= y
    # Same trick for D
    @views copyto!(H.scratchD[1:m-1], H.D[2:m])
    @views copyto!(H.D[1:m-1], H.scratchD[1:m-1])
    @views H.D[m:m] .= sTy
  end

  H.γ = sTy / yTy

  k = H.cur
  Sk = view(H.S, :, 1:k)
  Yk = view(H.Y, :, 1:k)
  mul!(view(H.StY, 1:k, 1:k), Sk', Yk)
  mul!(view(H.YtY, 1:k, 1:k), Yk', Yk)
  return H
end

"""
    apply_inv!(d, H::CompactInverseLBFGS, g, α = -one(T))

Compute `d .= α * H * g` in place using the compact representation.
The default `α = -1` produces the L-BFGS search direction `d = -H g`.
"""
function apply_inv!(d::AbstractVector{T}, H::CompactInverseLBFGS{T, V, M}, g::AbstractVector{T}, α::T = -one(T)) where {T, V, M}
  H.nprod += 1
  γ = H.γ
  k = H.cur
  if k == 0
    @. d = α * γ * g
    return d
  end

  Sk    = view(H.S, :, 1:k)
  Yk    = view(H.Y, :, 1:k)
  StYk  = view(H.StY, 1:k, 1:k)
  YtYk  = view(H.YtY, 1:k, 1:k)
  Dk    = view(H.D, 1:k)
  a1    = view(H.a1, 1:k)
  a2    = view(H.a2, 1:k)
  u1    = view(H.u1, 1:k)
  inner = view(H.inner, 1:k)

  # a1 = Sᵀ g, a2 = γ Yᵀ g
  mul!(a1, Sk', g)
  mul!(a2, Yk', g)
  a2 .*= γ

  # u1 = R⁻¹ a1  (R = upper triangle of StY, diag entries are Dk)
  u1 .= a1
  ldiv!(UpperTriangular(StYk), u1)

  # inner = γ (YᵀY) u1 + D u1 − a2
  mul!(inner, YtYk, u1)
  @. inner = γ * inner + Dk * u1 - a2

  # inner ← R⁻ᵀ inner = top block of M [a1; a2]
  # bottom block = −u1
  #
  # `UpperTriangular(StYk)'` evaluates to `LowerTriangular(adjoint(StYk))`,
  # and the generic `ldiv!(::AbstractTriangular, ::AbstractVecOrMat)` at
  # `stdlib/LinearAlgebra/src/triangular.jl:1177` calls `istriu(A)` to
  # pick the upper/lower branch. There's no `istriu(::LowerTriangular)`
  # specialization that short-circuits, so the generic implementation
  # iterates over the off-diagonal entries — fine on CPU, but on a
  # `CuArray`-backed view it triggers `Scalar indexing is disallowed`.
  # We already know we want the lower branch (`R'` is lower triangular by
  # construction), so we call `_ldiv!` directly and skip the check.
  let A = UpperTriangular(StYk)'   # ≡ `LowerTriangular(adjoint(StYk))`
    LinearAlgebra._ldiv!(inner, A, inner)
  end

  # d = γ g + S * inner + γ Y * (−u1) = γ g + S * inner − γ Y * u1
  @. d = γ * g
  mul!(d, Sk, inner, one(T), one(T))
  mul!(d, Yk, u1, -γ, one(T))

  d .*= α
  return d
end

"""
    compact_lbfgs(nlp; kwargs...)

A line-search limited-memory BFGS solver using the **compact** representation
of the inverse Hessian. Functionally equivalent to [`lbfgs`](@ref) but every
quasi-Newton product is a handful of BLAS-2 / triangular calls rather than
`mem` separate `dot`/`axpy` launches — substantially faster on GPU.

# Arguments
- `nlp::AbstractNLPModel{T, V}`: model to solve.
Keyword arguments mirror [`lbfgs`](@ref): `x`, `mem`, `atol`, `rtol`, `callback`,
`max_eval`, `max_time`, `max_iter`, `τ₁`, `bk_max`, `verbose`,
`verbose_subsolver`.

# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));
stats = compact_lbfgs(nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct CompactLBFGSSolver{T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, Model <: AbstractNLPModel{T, V}} <:
               AbstractOptimizationSolver
  x::V
  xt::V
  gx::V
  gt::V
  d::V
  H::CompactInverseLBFGS{T, V, M}
  h::LineModel{T, V, Model}
  params::CompactLBFGSParameterSet{T}
end

# Pick the matrix type that matches `V` (e.g. `Vector{T}` → `Matrix{T}`,
# `CuArray{T,1,…}` → `CuArray{T,2,…}`).
_matrix_type(x::AbstractVector{T}) where {T} = typeof(similar(x, T, 0, 0))

function CompactLBFGSSolver(nlp::Model; kwargs...) where {T, V, Model <: AbstractNLPModel{T, V}}
  nvar = nlp.meta.nvar

  params = CompactLBFGSParameterSet(nlp; kwargs...)
  mem = value(params.mem)

  x  = V(undef, nvar)
  d  = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)

  Mtype = _matrix_type(x)
  H = CompactInverseLBFGS(T, V, Mtype, nvar, mem)
  h = LineModel(nlp, x, d)
  return CompactLBFGSSolver{T, V, Mtype, Model}(x, xt, gx, gt, d, H, h, params)
end

function SolverCore.reset!(solver::CompactLBFGSSolver)
  LinearOperators.reset!(solver.H)
end

function SolverCore.reset!(solver::CompactLBFGSSolver, nlp::AbstractNLPModel)
  LinearOperators.reset!(solver.H)
  solver.h = LineModel(nlp, solver.x, solver.d)
  solver
end

@doc (@doc CompactLBFGSSolver) function compact_lbfgs(
  nlp::AbstractNLPModel{T, V};
  x::V = nlp.meta.x0,
  mem::Int = get(COMPACT_LBFGS_mem, nlp),
  τ₁::T = get(COMPACT_LBFGS_τ₁, nlp),
  bk_max::Int = get(COMPACT_LBFGS_bk_max, nlp),
  kwargs...,
) where {T, V}
  solver = CompactLBFGSSolver(nlp; mem = mem, τ₁ = τ₁, bk_max = bk_max)
  return solve!(solver, nlp; x = x, kwargs...)
end

function SolverCore.solve!(
  solver::CompactLBFGSSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  max_time::Float64 = 30.0,
  verbose::Int = 0,
  verbose_subsolver::Int = 0,
) where {T, V}
  if !(nlp.meta.minimize)
    error("compact_lbfgs only works for minimization problem")
  end
  if !unconstrained(nlp)
    error("compact_lbfgs should only be called for unconstrained problems. Try tron instead")
  end

  SolverCore.reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  τ₁     = value(solver.params.τ₁)
  bk_max = value(solver.params.bk_max)

  n = nlp.meta.nvar

  solver.x .= x
  x   = solver.x
  xt  = solver.xt
  ∇f  = solver.gx
  ∇ft = solver.gt
  d   = solver.d
  h   = solver.h
  H   = solver.H
  LinearOperators.reset!(H)

  f, ∇f = objgrad!(nlp, x, ∇f)

  ∇fNorm = nrm2(n, ∇f)
  ϵ = atol + rtol * ∇fNorm

  set_iter!(stats, 0)
  set_objective!(stats, f)
  set_dual_residual!(stats, ∇fNorm)

  verbose > 0 && @info log_header(
    [:iter, :f, :dual, :slope, :bk],
    [Int, T, T, T, Int],
    hdr_override = Dict(:f => "f(x)", :dual => "‖∇f‖", :slope => "∇fᵀd"),
  )
  verbose > 0 && @info log_row(Any[stats.iter, f, ∇fNorm, T, Int])

  optimal = ∇fNorm ≤ ϵ
  fmin = min(-one(T), f) / eps(T)
  unbounded = f < fmin

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
      unbounded = unbounded,
      max_eval = max_eval,
      iter = stats.iter,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done
    apply_inv!(d, H, ∇f, -one(T))
    slope = dot(n, d, ∇f)
    if slope ≥ 0
      @error "not a descent direction" slope
      set_status!(stats, :not_desc)
      done = true
      continue
    end

    t, good_grad, ft, nbk, nbW =
      armijo_wolfe(h, f, slope, ∇ft, τ₁ = τ₁, bk_max = bk_max, verbose = Bool(verbose_subsolver))

    copyaxpy!(n, t, d, x, xt)
    good_grad || grad!(nlp, xt, ∇ft)

    # Build the curvature pair (s, y) = (t*d, ∇ft - ∇f) and push it into H.
    d .*= t
    @. ∇f = ∇ft - ∇f
    push!(H, d, ∇f)

    x .= xt
    f = ft
    ∇f .= ∇ft

    ∇fNorm = nrm2(n, ∇f)

    set_iter!(stats, stats.iter + 1)

    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row(Any[stats.iter, f, ∇fNorm, slope, nbk])

    set_objective!(stats, f)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, ∇fNorm)
    optimal = ∇fNorm ≤ ϵ
    unbounded = f < fmin

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        unbounded = unbounded,
        max_eval = max_eval,
        iter = stats.iter,
        max_iter = max_iter,
        max_time = max_time,
      ),
    )
    set_solver_specific!(stats, :nprod, solver.H.nprod)

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end
  verbose > 0 && @info log_row(Any[stats.iter, f, ∇fNorm])

  set_solution!(stats, x)
  stats
end
