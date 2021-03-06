module DiffOperatorsTests

using Test
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Fields: MockField
using LinearAlgebra
using FillArrays

np = 4
p = Point(1,2)
x = fill(p,np)

v = 3.0
d = 2
_f = MockField{d}(v)

l = 10
_af = Fill(_f,l)

for f in (_f,_af)

  @test ∇(f) == gradient(f)
  
  @test divergence(f) == tr(gradient(f))
  
  @test curl(f) == grad2curl(gradient(f))
  
  @test ∇*f == divergence(f)
  
  @test cross(∇,f) == curl(f)
  
  @test outer(∇,f) == ∇(f)
  
  @test outer(f,∇) == transpose(∇(f))

  @test ε(f) == symmetric_part(gradient(f))

  @test Δ(f) == ∇*∇(f)

end

# Test automatic differentiation

u_scal(x) = x[1]^2 + x[2]
∇u_scal(x) = VectorValue( 2*x[1], one(x[2]) )
Δu_scal(x) = 2

u_vec(x) = VectorValue( x[1]^2 + x[2], 4*x[1] - x[2]^2 )
∇u_vec(x) = TensorValue( 2*x[1], one(x[2]), 4*one(x[1]), - 2*x[2] )
Δu_vec(x) = VectorValue( 2, -2 )

xs = [ Point(1.,1.), Point(2.,0.), Point(0.,3.), Point(-1.,3.)]
for x in xs
  @test ∇(u_scal)(x) == ∇u_scal(x)
  @test Δ(u_scal)(x) == Δu_scal(x)
  @test ∇(u_vec)(x) == ∇u_vec(x)
  @test Δ(u_vec)(x) == Δu_vec(x)
end

u(x) = VectorValue( x[1]^2 + 2*x[2]^2, -x[1]^2 )
∇u(x) = TensorValue( 2*x[1], 4*x[2], -2*x[1], zero(x[1]) )
Δu(x) = VectorValue( 6, -2 )

for x in xs
  @test (∇*u)(x) == tr(∇u(x)) 
  @test (∇×u)(x) == grad2curl(∇u(x))
  @test Δ(u)(x) == Δu(x)
end

end # module
