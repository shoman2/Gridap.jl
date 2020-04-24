using Test
using Gridap
import Gridap: ∇, divergence
using Gridap.Geometry


domain = (0,1,0,1)
partition = (4,4)
model = CartesianDiscreteModel(domain,partition)
order = 2

trian = get_triangulation(model)
degree = order
quad = CellQuadrature(trian,degree)

u(x) = VectorValue(x[1]^2,x[2])
f(x) = - Δ(u)(x)

labels = get_face_labeling(model)

V = TestFESpace(
   model=model,
   order=order,
   reffe=:Lagrangian,
   labels=labels,
   valuetype=VectorValue{2,Float64},
   dirichlet_tags="boundary")

U = TrialFESpace(V,u)

uh = interpolate(U,u)

a(u,v) = inner(∇(u),∇(v))
l(v) = v*f
t_Ω = AffineFETerm(a,l,trian,quad)

op = AffineFEOperator(U,V,t_Ω)

ls = LUSolver()
solver = LinearFESolver(ls)
uh = solve(solver,op)
l2(u) = inner(u,u)
sh1(u) = a(u,u)
h1(u) = sh1(u) + l2(u)
e = u - uh
el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))
ul2 = sqrt(sum( integrate(l2(uh),trian,quad) ))
uh1 = sqrt(sum( integrate(h1(uh),trian,quad) ))

@test el2/ul2 < 1.e-8
@test eh1/uh1 < 1.e-7
