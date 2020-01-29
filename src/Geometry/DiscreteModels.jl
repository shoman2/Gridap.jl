"""
    abstract type DiscreteModel{Dc,Dp} <: GridapType

Abstract type holding information about a physical grid, 
the underlying grid topology, and a labeling of
the grid faces. This is the information that typically provides a mesh
generator, and it is what one needs to perform a simulation.

The `DiscreteModel` interface is defined by overloading the methods:

- [`get_grid(model::DiscreteModel)`](@ref)
- [`get_grid_topology(model::DiscreteModel)`](@ref)
- [`get_face_labeling(g::DiscreteModel)`](@ref)

The interface is tested with this function:

- [`test_discrete_model`](@ref)

"""
abstract type DiscreteModel{Dc,Dp} <: GridapType end

"""
    get_grid(model::DiscreteModel)
"""
function get_grid(model::DiscreteModel)
  @abstractmethod
end

"""
    get_grid_topology(model::DiscreteModel)
"""
function get_grid_topology(model::DiscreteModel)
  @abstractmethod
end

"""
    get_face_labeling(g::DiscreteModel)
"""
function get_face_labeling(g::DiscreteModel)
  @abstractmethod
end

# Testers

"""
    test_discrete_model(model::DiscreteModel)
"""
function test_discrete_model(model::DiscreteModel{Dc,Dp}) where {Dc,Dp}
  D = Dc
  grid = get_grid(model)
  test_grid(grid)
  topo = get_grid_topology(model)
  test_grid_topology(topo)
  @test num_cell_dims(topo) == Dc
  @test num_point_dims(topo) == Dp
  labels = get_face_labeling(model)
  @test num_faces(labels) == num_faces(topo)
  for d in 0:D
    @test num_faces(labels,d) == num_faces(topo,d)
  end
end

# Delegators to the underlying grid

get_cell_nodes(g::DiscreteModel) = get_cell_nodes(get_grid(g))

get_node_coordinates(g::DiscreteModel) = get_node_coordinates(get_grid(g))

get_cell_type(g::DiscreteModel) = get_cell_type(get_grid(g))

get_reffes(g::DiscreteModel) = get_reffes(get_grid(g))

# Default API

"""
    num_dims(model::DiscreteModel)
"""
num_dims(model::DiscreteModel) = num_dims(get_grid_topology(model))

"""
    num_cell_dims(model::DiscreteModel)
"""
num_cell_dims(model::DiscreteModel) = num_cell_dims(get_grid_topology(model))

"""
    num_point_dims(model::DiscreteModel)
"""
num_point_dims(model::DiscreteModel) = num_point_dims(get_grid_topology(model))

"""
    num_faces(g::DiscreteModel,d::Integer)
    num_faces(g::DiscreteModel)
"""
num_faces(g::DiscreteModel,d::Integer) = num_faces(get_grid_topology(g),d)
num_faces(g::DiscreteModel) = num_faces(get_grid_topology(g))

"""
    num_cells(g::DiscreteModel)
"""
num_cells(g::DiscreteModel) = num_cells(get_grid_topology(g))

"""
    num_facets(g::DiscreteModel)
"""
num_facets(g::DiscreteModel) = num_facets(get_grid_topology(g))

"""
    num_edges(g::DiscreteModel)
"""
num_edges(g::DiscreteModel) = num_edges(get_grid_topology(g))

"""
    num_vertices(g::DiscreteModel)
"""
num_vertices(g::DiscreteModel) = num_vertices(get_grid_topology(g))

"""
    num_nodes(g::DiscreteModel)
"""
num_nodes(g::DiscreteModel) = num_nodes(get_grid(g))

"""
    get_polytopes(model::DiscreteModel)
"""
function get_polytopes(model::DiscreteModel)
  topo = get_grid_topology(model)
  get_polytopes(topo)
end

"""
    get_face_nodes(g::DiscreteModel,d::Integer)
"""
function get_face_nodes(g::DiscreteModel,d::Integer)
  compute_face_nodes(g,d)
end

"""
    get_face_nodes(g::DiscreteModel)
"""
function get_face_nodes(g::DiscreteModel)
  compute_face_nodes(g)
end

"""
    compute_face_nodes(model::DiscreteModel,d::Integer)
"""
function compute_face_nodes(model::DiscreteModel,d::Integer)

  if d == num_cell_dims(model)
    return get_cell_nodes(model)
  end

  topo = get_grid_topology(model)
  D = num_cell_dims(topo)
  cell_to_nodes = Table(get_cell_nodes(model))
  cell_to_faces = Table(get_faces(topo,D,d))
  cell_to_ctype = get_cell_type(model)
  reffes = get_reffes(model)
  ctype_to_lface_to_lnodes = map( (reffe)-> get_face_nodes(reffe,d) , reffes )
  nfaces = num_faces(topo,d)

  face_to_nodes = generate_face_to_vertices(
    cell_to_nodes,
    cell_to_faces,
    cell_to_ctype,
    ctype_to_lface_to_lnodes,
    nfaces)

  face_to_nodes

end

"""
    compute_face_nodes(model::DiscreteModel)
"""
function compute_face_nodes(model::DiscreteModel)
  D = num_cell_dims(model)
  data = [compute_face_nodes(model,d) for d in 0:D]
  append_tables_globally(data...)
end


"""
    get_face_own_nodes(g::DiscreteModel,d::Integer)
"""
function get_face_own_nodes(g::DiscreteModel,d::Integer)
  compute_face_own_nodes(g,d)
end

"""
    get_face_own_nodes(g::DiscreteModel)
"""
function get_face_own_nodes(g::DiscreteModel)
  compute_face_own_nodes(g)
end

"""
    compute_face_own_nodes(model::DiscreteModel,d::Integer)
"""
function compute_face_own_nodes(model::DiscreteModel,d::Integer)

  topo = get_grid_topology(model)
  D = num_cell_dims(topo)
  cell_to_nodes = Table(get_cell_nodes(model))
  cell_to_faces = Table(get_faces(topo,D,d))
  cell_to_ctype = get_cell_type(model)
  reffes = get_reffes(model)
  ctype_to_lface_to_lnodes = map( (reffe)-> get_face_own_nodes(reffe,d) , reffes )
  nfaces = num_faces(topo,d)

  face_to_own_nodes = generate_face_to_vertices(
    cell_to_nodes,
    cell_to_faces,
    cell_to_ctype,
    ctype_to_lface_to_lnodes,
    nfaces)

  face_to_own_nodes

end

"""
    compute_face_own_nodes(model::DiscreteModel)
"""
function compute_face_own_nodes(model::DiscreteModel)
  D = num_cell_dims(model)
  data = [compute_face_own_nodes(model,d) for d in 0:D]
  append_tables_globally(data...)
end

"""
    get_vertex_node(g::DiscreteModel)
"""
function get_vertex_node(g::DiscreteModel)
  compute_vertex_node(g)
end

"""
    compute_vertex_node(g::DiscreteModel)
"""
function compute_vertex_node(g::DiscreteModel)
  d=0
  vertex_to_nodes = Table(get_face_own_nodes(g,d))
  vertex_to_nodes.data
end

"""
    get_node_face_owner(g::DiscreteModel)
"""
function get_node_face_owner(g::DiscreteModel)
  compute_node_face_owner(g)
end

"""
    compute_node_face_owner(g::DiscreteModel)
"""
function compute_node_face_owner(g::DiscreteModel)
  face_to_own_nodes = Table(get_face_own_nodes(g))
  node_to_face_owner = zeros(Int,num_nodes(g))
  _compute_node_face_owner!(node_to_face_owner,face_to_own_nodes)
  node_to_face_owner
end

function  _compute_node_face_owner!(node_to_face_owner,face_to_own_nodes)
  for face in 1:length(face_to_own_nodes)
    pini = face_to_own_nodes.ptrs[face]
    pend = face_to_own_nodes.ptrs[face+1]-1
    for p in pini:pend
      node = face_to_own_nodes.data[p]
      node_to_face_owner[node] = face
    end
  end
end

"""
    get_reffaces(::Type{ReferenceFE{d}},model::DiscreteModel) where d
"""
function get_reffaces(::Type{ReferenceFE{d}},model::DiscreteModel) where d
  reffaces,_ = compute_reffaces(ReferenceFE{d},model)
  reffaces
end

"""
    get_face_type(g::DiscreteModel,d::Integer)

Index to the vector `get_reffaces(ReferenceFE{d},g)`
"""
function get_face_type(g::DiscreteModel,d::Integer)
  _, face_to_ftype = compute_reffaces(ReferenceFE{d},g)
  face_to_ftype
end

"""
    compute_reffaces(::Type{ReferenceFE{d}}, g::DiscreteModel) where d
"""
function compute_reffaces(::Type{ReferenceFE{d}}, g::DiscreteModel) where d
  D = num_cell_dims(g)
  topo = get_grid_topology(g)
  ctype_to_reffe = get_reffes(g)
  ctype_to_lftype_to_refface = [ get_reffaces(ReferenceFE{d},reffe) for reffe in ctype_to_reffe]
  ctype_to_lface_to_lftype = [ get_face_type(reffe,d) for reffe in ctype_to_reffe]
  t = _generate_ftype_to_refface(Val{d}(),ctype_to_lftype_to_refface,ctype_to_lface_to_lftype)
  ftype_to_refface, ctype_to_lface_to_ftype = t
  cell_to_faces = Table(get_faces(topo,D,d))
  cell_to_ctype = get_cell_type(g)
  nfaces = num_faces(g,d)
  face_to_ftype = generate_face_to_face_type(
    cell_to_faces, cell_to_ctype, ctype_to_lface_to_ftype, nfaces)
  (collect1d(ftype_to_refface), face_to_ftype)
end

function compute_reffaces(::Type{ReferenceFE{D}}, g::DiscreteModel{D}) where D
  (get_reffes(g), get_cell_type(g))
end

"""
    get_reffaces(model::DiscreteModel)
"""
function get_reffaces(model::DiscreteModel)
  reffaces, _ , _ = compute_reffaces(model)
  reffaces
end

"""
    get_face_type(model::DiscreteModel)
"""
function get_face_type(model::DiscreteModel)
  _, face_to_ftype, _ = compute_reffaces(model)
  face_to_ftype
end

"""
    get_reffaces_offsets(model::DiscreteModel)
"""
function get_reffaces_offsets(model::DiscreteModel)
  _, _, offsets = compute_reffaces(model)
  offsets
end

"""
    compute_reffaces(g::DiscreteModel)
"""
function compute_reffaces(g::DiscreteModel)
  D = num_cell_dims(g)
  d_to_refdfaces = Vector{LagrangianRefFE}[]
  d_to_dface_to_ftype = Vector{Int8}[]
  for d in 0:D
    push!(d_to_refdfaces,get_reffaces(ReferenceFE{d},g))
    push!(d_to_dface_to_ftype,get_face_type(g,d))
  end
  d_to_offset = zeros(Int,D+1)
  for d in 1:D
    d_to_offset[d+1] = d_to_offset[d] + length(d_to_refdfaces[d])
    d_to_dface_to_ftype[d+1] .+= d_to_offset[d+1]
  end
  (vcat(d_to_refdfaces...), vcat(d_to_dface_to_ftype...), d_to_offset)
end

"""
    Grid(::Type{ReferenceFE{d}},model::DiscreteModel) where d
"""
function Grid(::Type{ReferenceFE{d}},model::DiscreteModel) where d
  node_coordinates = collect1d(get_node_coordinates(model))
  cell_to_nodes = Table(get_face_nodes(model,d))
  cell_to_type = collect1d(get_face_type(model,d))
  reffes = get_reffaces(ReferenceFE{d},model)
  UnstructuredGrid(node_coordinates, cell_to_nodes, reffes, cell_to_type)
end

function Grid(::Type{ReferenceFE{d}},model::DiscreteModel{d}) where d
  get_grid(model)
end

"""
    Triangulation(::Type{ReferenceFE{d}},model::DiscreteModel) where d
"""
function Triangulation(::Type{ReferenceFE{d}},model::DiscreteModel) where d
  Grid(ReferenceFE{d},model)
end

"""
    get_triangulation(model::DiscreteModel)
"""
function get_triangulation(model::DiscreteModel)
  get_grid(model)
end

"""
    simplexify(model::DiscreteModel)
"""
function simplexify(model::DiscreteModel)
  simplexify(UnstructuredDiscreteModel(model))
end

# IO

function to_dict(model::DiscreteModel)
  umodel = UnstructuredDiscreteModel(model)
  to_dict(umodel)
end

function from_dict(::Type{DiscreteModel},dict::Dict{Symbol,Any})
  from_dict(UnstructuredDiscreteModel,dict)
end
