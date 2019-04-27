module MapsTests
##
using Numa
using Test
using Numa.Maps
using Numa.FieldValues

import Numa: evaluate, gradient
import Numa: evaluate!, return_size
import Base: +, -, *, /, ∘
import Numa.FieldValues: inner, outer

include("MockMap.jl")

a = Point{2}(10,10)
b = Point{2}(15,20)
p1 = Point{2}(1,1)
p2 = Point{2}(2,2)
p3 = Point{2}(3,3)
p = [p1,p2,p3]
##
@testset "MockMap" begin
  length(p)
  map = MockMap(a)
  res = evaluate(map,p)
  for i in 1:length(p)
    @test res[i] == a+p[i]
  end
  @test return_size(map,size(p)) == size(p)
  gmap = gradient(map)
  gres = evaluate(gmap,p)
  for i in 1:length(p)
    @test gres[i] == p[i]
  end
end

# Unary Operators
mymap = MockMap(a)
using Numa.Maps: MapFromUnaryOp
res = evaluate(mymap,p)
@testset "UnaryOp" begin
  for op in (:+, :-)
    @eval begin
      umap = MapFromUnaryOp($op,mymap)
      res2 = evaluate(umap,p)
      for i in 1:length(p)
        @test res2[i] == $op(res[i])
      end
      isa(umap,MapFromUnaryOp{typeof($op),MockMap{2}})
    end
  end
end

# Binary Operators
map1 = MockMap(a)
map2 = MockMap(b)
res1 = evaluate(map1,p)
res2 = evaluate(map2,p)
using Numa.Maps: MapFromBinaryOp
@testset "BinaryOp" begin
  for op in (:+, :-, :inner, :outer)
    @eval begin
      umap = MapFromBinaryOp($op,map1,map2)
      resu = evaluate(umap,p)
      for i in 1:length(p)
        @test resu[i] == $op(res1[i],res2[i])
      end
      isa(umap,MapFromBinaryOp{typeof($op),MockMap{2}})
    end
  end
end

# Compose
f(p::Point{2}) = 2*p
gradf(p::Point{2}) = VectorValue(2.0,2.0)
gradient(::typeof(f)) = gradf
@testset "ComposeField" begin
  using Numa.Maps: FieldFromCompose
  @test MockMap <: Field
  map = MockMap(a)
  umap = FieldFromCompose(f,map)
  res = evaluate(map,p)
  resu = evaluate(umap,p)
  for i in 1:length(p)
    @test resu[i] == f(res[i])
  end
  gumap = gradient(umap)
  gres = evaluate(gumap,p)
  for i in 1:length(p)
    @test gres[i] == gradf(p[i])
  end
end



# FieldFromExpand

# FieldFromComposeExtended
using Numa.Maps: Geomap
MockMap <: Geomap
map = MockMap(a)
geomap = MockMap(b)
@testset "ComposeExtended" begin
  using Numa.Maps: FieldFromComposeExtended
  cemap = FieldFromComposeExtended(f,geomap,map)
  res = evaluate(cemap,p)
  for i in 1:length(p)
    res[i] == f(evaluate(map,evaluate(geomap,[p[i]]))...)
  end
  gcemap = gradient(cemap)
  gres = gradient(cemap)
  gres = evaluate(gcemap,p)
  for i in 1:length(p)
    @test gres[i] == gradf(p[i])
  end
end

end # module Maps
