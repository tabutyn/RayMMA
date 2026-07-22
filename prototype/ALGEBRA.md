# Separated ray/triangle intersection algebra

This note derives the exact real-arithmetic expressions approximated by the
accompanying WMMA prototype. The purpose is to ensure that every
multiplication performed for a ray/triangle pair combines a **triangle-only
coefficient** with a **ray-only feature**.

## 1. Geometry convention

Write the triangle as

```text
P(u,v) = A + u C + v B
```

where `A` is a vertex, `C` is the edge associated with barycentric coordinate
`u`, and `B` is the edge associated with `v`. Therefore the vertices are
`A`, `A+C`, and `A+B`. If the input contains three endpoint positions instead,
first compute the two edge vectors.

Write the ray as

```text
R(t) = O + t r
```

The intersection system is

```text
O + t r = A + u C + v B.
```

Let `q = A-O`. Rearrangement gives the 3-by-3 system

```text
[ r  -C  -B ] [ t u v ]^T = q.
```

## 2. Cramer's-rule result

Define the shared determinant

```text
Delta = dot(r, cross(C,B)).
```

The solution is

```text
Nt = dot(A-O, cross(C,B))
Nu = dot(r, cross(B,A-O))
Nv = dot(r, cross(A-O,C))

t = Nt / Delta
u = Nu / Delta
v = Nv / Delta.
```

All three coordinates share the same denominator. `Delta == 0` means that the
ray and triangle plane are parallel (or that the triangle is degenerate).

## 3. Separate the triangle and ray variables

Precompute these values for each triangle:

```text
n   = cross(C,B)
an  = dot(A,n)
pu  = cross(B,A)
pv  = cross(A,C)
```

Precompute the ray moment once for each ray:

```text
m = cross(O,r).
```

Expanding the Cramer's-rule numerators and applying scalar-triple-product
identities gives

```text
Nt    = an - dot(n,O)
Nu    = dot(pu,r) - dot(B,m)
Nv    = dot(pv,r) + dot(C,m)
Delta = dot(n,r).
```

The triangle and ray variables are now completely separated. In particular,
`(r,m)` are the six Plucker-style coordinates of the ray's supporting line.

## 4. Feature-vector form

Use the ten-component ray feature vector

```text
phi = [ 1, Ox, Oy, Oz, rx, ry, rz, mx, my, mz ]^T.
```

For each triangle, construct four coefficient rows. Vertical bars below only
show the feature groups `1 | O | r | m`:

```text
Ktriangle = [ an | -n |  0 |  0 ]   -> Nt
            [  0 |  0 | pu | -B ]   -> Nu
            [  0 |  0 | pv |  C ]   -> Nv
            [  0 |  0 |  n |  0 ]   -> Delta
```

Thus

```text
Ktriangle * phi = [ Nt, Nu, Nv, Delta ]^T.
```

The CUDA implementation pads both operands with six zeros to obtain a
16-component inner dimension. Four triangles contribute 16 coefficient rows,
while 16 rays contribute 16 columns. One `m16n16k16` WMMA tile therefore
produces the four values for each of 4*16 = 64 ray/triangle pairs.

### Direct three-edge row form

An alternative tile evaluates all three oriented edge functions directly.
Let `V0=A`, `V1=A+C`, `V2=A+B`, and retain `m=cross(O,r)`. Use:

```text
E0 = dot(cross(V1,V2),r) + dot(V2-V1,m)
E1 = dot(cross(V2,V0),r) + dot(V0-V2,m)
E2 = dot(cross(V0,V1),r) + dot(V1-V0,m)
```

Then `Delta=E0+E1+E2`, `u=E1/Delta`, and `v=E2/Delta`. A fourth `Nt` row
retains depth, yielding the row order `[E0,E1,E2,Nt]`. This form makes the
third edge an independent Tensor result rather than subtracting `Nu` and `Nv`
from `Delta`.

## 5. Hit testing without immediate division

For a two-sided triangle, first orient all values to a positive denominator:

```text
s  = sign(Delta)
d  = s * Delta
T  = s * Nt
U  = s * Nu
V  = s * Nv
```

Ignoring a small parallel-ray tolerance, the candidate is inside the triangle
and inside `[tmin,tmax]` when

```text
d > 0
U >= 0
V >= 0
U + V <= d
T >= tmin * d
T <= tmax * d.
```

Only surviving candidates need divisions. The recovered coordinates are
`t=T/d`, `u=U/d`, and `v=V/d`.

## 6. Numerical considerations

- WMMA inputs in this prototype are FP16 and accumulators are FP32.
- Compute cross products in FP32 before conversion to FP16.
- Large world-space positions make `cross(O,r)`, `cross(B,A)`, and `dot(A,n)`
  vulnerable to FP16 overflow and cancellation. The integrated implementation
  uses BVH-leaf-local coordinates; this algebra prototype does not.
- If a local transform is `P' = s(P-center)` while `r` remains unscaled, the
  Tensor ratio returns `t' = s*t`; recover world depth with `t=t'/s`. Edge
  values scale as `s^2`, while `Nt` scales as `s^3`.
- Multiplying all four rows of one triangle by the same power of two preserves
  every exact ratio before quantization and can keep coefficients for triangles
  smaller than their shared four-triangle frame out of FP16's
  subnormal/underflow range. It does not add relative mantissa precision to
  values already normal in FP16.
- Use an FP32 fallback near `Delta == 0` and near barycentric boundaries if
  watertight behavior is required.
- The method evaluates a dense Cartesian product. It is useful only when the
  traversal stage supplies reasonably full packets of candidate triangles and
  rays.
