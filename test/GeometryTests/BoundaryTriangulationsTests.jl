module BoundaryTriangulationsTests

using Test
using Gridap.Helpers
using Gridap.Fields
using Gridap.Arrays
using Gridap.ReferenceFEs
using Gridap.Geometry

domain = (0,4,0,4)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

btrian = BoundaryTriangulation(model)
test_boundary_triangulation(btrian)

cellids = collect(1:num_cells(model))

face_to_cellid = reindex(cellids,btrian)
@test face_to_cellid == get_face_to_cell(btrian)

trian = get_volume_triangulation(btrian)
q2x = get_cell_map(trian)
s2x = restrict(q2x,btrian)

s = CompressedArray([Point{1,Float64}[(0.25,),(0.75,)]],get_cell_type(btrian))
x = evaluate(s2x,s)
r = Vector{Point{2,Float64}}[
  [(0.5,0.0),(1.5,0.0)],[(0.0,0.5),(0.0,1.5)],
  [(2.5,0.0),(3.5,0.0)],[(4.0,0.5),(4.0,1.5)],
  [(0.5,4.0),(1.5,4.0)],[(0.0,2.5),(0.0,3.5)],
  [(2.5,4.0),(3.5,4.0)],[(4.0,2.5),(4.0,3.5)]]
test_array(x,r)

nvec = get_normal_vector(btrian)
nvec_x = evaluate(nvec,s)

domain = (0,4,0,4)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

btrian = BoundaryTriangulation(model,"tag_8")
test_boundary_triangulation(btrian)

btrian = BoundaryTriangulation(model,get_face_labeling(model),"tag_8")
test_boundary_triangulation(btrian)

s = CompressedArray([Point{1,Float64}[(0.25,),(0.75,)]],get_cell_type(btrian))
nvec = get_normal_vector(btrian)
nvec_x = evaluate(nvec,s)
s2x = get_cell_map(btrian)
x = evaluate(s2x,s)

@test get_cell_id(btrian) == get_face_to_cell(btrian)
r = rand(num_cells(trian))
@test reindex(r,btrian) == r[get_face_to_cell(btrian)]

end # module
