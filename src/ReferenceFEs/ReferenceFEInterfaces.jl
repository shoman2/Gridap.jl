"""
    abstract type ReferenceFE{D} <: GridapType

Abstract type representing a Reference finite element. `D` is the underlying coordinate space dimension.
We follow the Ciarlet definition. A reference finite element
is defined by a polytope (cell topology), a basis of an interpolation space
of top of this polytope (denoted here as the prebasis), and a basis of the dual of this space
(i.e. the degrees of freedom). From this information one can compute the shape functions
(i.e, the canonical basis of w.r.t. the degrees of freedom) with a simple change of basis.
In addition, we also encode in this type information about how the interpolation space
in a reference finite element is "glued" with neighbors in order to build conforming
cell-wise spaces.

The `ReferenceFE` interface is defined by overloading these methods:

- [`num_dofs(reffe::ReferenceFE)`](@ref)
- [`get_polytope(reffe::ReferenceFE)`](@ref)
- [`get_prebasis(reffe::ReferenceFE)`](@ref)
- [`get_dof_basis(reffe::ReferenceFE)`](@ref)
- [`get_face_own_dofs(reffe::ReferenceFE)`](@ref)
- [`get_face_own_dofs_permutations(reffe::ReferenceFE)`](@ref)
- [`get_face_dofs(reffe::ReferenceFE)`](@ref)

The interface is tested with
- [`test_reference_fe(reffe::ReferenceFE)`](@ref)

"""
abstract type ReferenceFE{D} <: GridapType end

"""
    num_dofs(reffe::ReferenceFE) -> Int

Returns the number of DOFs.
"""
function num_dofs(reffe::ReferenceFE)
  @abstractmethod
end

"""
    get_polytope(reffe::ReferenceFE) -> Polytope

Returns the underlying polytope object.
"""
function get_polytope(reffe::ReferenceFE)
  @abstractmethod
end

"""
    get_prebasis(reffe::ReferenceFE) -> Field

Returns the underlying prebasis encoded as a `Field` object.
"""
function get_prebasis(reffe::ReferenceFE)
  @abstractmethod
end

"""
    get_dof_basis(reffe::ReferenceFE) -> Dof

Returns the underlying dof basis encoded in a `Dof` object. 
"""
function get_dof_basis(reffe::ReferenceFE)
  @abstractmethod
end

"""
    get_face_own_dofs(reffe::ReferenceFE) -> Vector{Vector{Int}}
"""
function get_face_own_dofs(reffe::ReferenceFE)
  @abstractmethod
end

"""
    get_face_own_dofs_permutations(reffe::ReferenceFE) -> Vector{Vector{Vector{Int}}}
"""
function get_face_own_dofs_permutations(reffe::ReferenceFE)
  @abstractmethod
end

"""
    get_face_dofs(reffe::ReferenceFE) -> Vector{Vector{Int}}

Returns a vector of vector that, for each face, stores the
dofids in the closure of the face.
"""
function get_face_dofs(reffe::ReferenceFE)
  @abstractmethod
end

# Test

"""
    test_reference_fe(reffe::ReferenceFE{D}) where D

Test if the methods in the `ReferenceFE` interface are defined for the object `reffe`.
"""
function test_reference_fe(reffe::ReferenceFE{D}) where D
  @test D == num_dims(reffe)
  p = get_polytope(reffe)
  @test isa(p,Polytope{D})
  basis = get_prebasis(reffe)
  @test isa(basis,Field)
  dofs = get_dof_basis(reffe)
  @test isa(dofs,Dof)
  facedofs = get_face_own_dofs(reffe)
  @test isa(facedofs,Vector{Vector{Int}})
  @test length(facedofs) == num_faces(p)
  facedofs_perms = get_face_own_dofs_permutations(reffe)
  @test isa(facedofs_perms,Vector{Vector{Vector{Int}}})
  @test length(facedofs_perms) == num_faces(p)
  facedofs = get_face_dofs(reffe)
  @test isa(facedofs,Vector{Vector{Int}})
  @test length(facedofs) == num_faces(p)
  shapefuns = get_shapefuns(reffe)
  @test isa(shapefuns,Field)
  ndofs = num_dofs(reffe)
  m = evaluate(dofs,basis)
  @test ndofs == size(m,1)
  @test ndofs == size(m,2)
end


"""
Constant of type `Int`  used to signal that a permutation is not valid.
"""
const INVALID_PERM = 0

# API

"""
    num_dims(::Type{<:ReferenceFE{D}}) where D
    num_dims(reffe::ReferenceFE{D}) where D

Returns `D`.
"""
num_dims(reffe::ReferenceFE) = num_dims(typeof(reffe))
num_dims(::Type{<:ReferenceFE{D}}) where D = D

"""
    num_cell_dims(::Type{<:ReferenceFE{D}}) where D
    num_cell_dims(reffe::ReferenceFE{D}) where D

Returns `D`.
"""
num_cell_dims(reffe::ReferenceFE) = num_dims(typeof(reffe))
num_cell_dims(::Type{<:ReferenceFE{D}}) where D = D

"""
    num_point_dims(::Type{<:ReferenceFE{D}}) where D
    num_point_dims(reffe::ReferenceFE{D}) where D

Returns `D`.
"""
num_point_dims(reffe::ReferenceFE) = num_dims(typeof(reffe))
num_point_dims(::Type{<:ReferenceFE{D}}) where D = D

"""
    num_faces(reffe::ReferenceFE)
    num_faces(reffe::ReferenceFE,d::Integer)
"""
num_faces(reffe::ReferenceFE) = num_faces(get_polytope(reffe))
num_faces(reffe::ReferenceFE,d::Integer) = num_faces(get_polytope(reffe),d)

"""
    num_vertices(reffe::ReferenceFE)
"""
num_vertices(reffe::ReferenceFE) = num_vertices(get_polytope(reffe))


"""
    num_edges(reffe::ReferenceFE)
"""
num_edges(reffe::ReferenceFE) = num_edges(get_polytope(reffe))


"""
    num_facets(reffe::ReferenceFE)
"""
num_facets(reffe::ReferenceFE) = num_facets(get_polytope(reffe))


"""
    get_face_own_dofs(reffe::ReferenceFE,d::Integer)
"""
function get_face_own_dofs(reffe::ReferenceFE,d::Integer)
  p = get_polytope(reffe)
  range = get_dimrange(p,d)
  get_face_own_dofs(reffe)[range]
end

"""
    get_face_dofs(reffe::ReferenceFE,d::Integer)
"""
function get_face_dofs(reffe::ReferenceFE,d::Integer)
  p = get_polytope(reffe)
  range = get_dimrange(p,d)
  get_face_dofs(reffe)[range]
end

"""
    get_face_own_dofs_permutations(reffe::ReferenceFE,d::Integer)
"""
function get_face_own_dofs_permutations(reffe::ReferenceFE,d::Integer)
  p = get_polytope(reffe)
  range = get_dimrange(p,d)
  get_face_own_dofs_permutations(reffe)[range]
end

"""
    get_own_dofs_permutations(reffe::ReferenceFE)
"""
function get_own_dofs_permutations(reffe::ReferenceFE)
  n = num_faces(get_polytope(reffe))
  get_face_own_dofs_permutations(reffe)[n]
end

"""
    get_shapefuns(reffe::ReferenceFE) -> Field

Returns the basis of shape functions (i.e. the canonical basis)
associated with the reference FE. The result is encoded as a `Field` object.
"""
function get_shapefuns(reffe::ReferenceFE)
  dofs = get_dof_basis(reffe)
  prebasis = get_prebasis(reffe)
  compute_shapefuns(dofs,prebasis)
end

"""
    compute_shapefuns(dofs,prebasis)

Helper function used to compute the shape function basis
associated with the dof basis `dofs` and the basis `prebasis`.

It is equivalent to

    change = inv(evaluate(dofs,prebasis))
    change_basis(prebasis,change)
"""
function compute_shapefuns(dofs,prebasis)
  change = inv(evaluate(dofs,prebasis))
  change_basis(prebasis,change)
end

# Concrete implementation

"""
    struct GenericRefFE{D} <: ReferenceFE{D}
      ndofs::Int
      polytope::Polytope{D}
      prebasis::Field
      dofs::Dof
      face_own_dofs::Vector{Vector{Int}}
      face_own_dofs_permutations::Vector{Vector{Vector{Int}}}
      face_dofs::Vector{Vector{Int}}
      shapefuns::Field
    end

This type is a *materialization* of the `ReferenceFE` interface. That is, it is a 
`struct` that stores the values of all abstract methods in the `ReferenceFE` interface.
This type is useful to build reference FEs from the underlying ingredients without
the need to create a new type.

Note that some fields in this `struct` are type unstable deliberately in order to simplify the
type signature. Don't access them in computationally expensive functions,
instead extract the required fields before and pass them to the computationally expensive function.
"""
struct GenericRefFE{D} <: ReferenceFE{D}
  ndofs::Int
  polytope::Polytope{D}
  prebasis::Field
  dofs::Dof
  face_own_dofs::Vector{Vector{Int}}
  face_own_dofs_permutations::Vector{Vector{Vector{Int}}}
  face_dofs::Vector{Vector{Int}}
  shapefuns::Field
  @doc """
      GenericRefFE(
        ndofs::Int,
        polytope::Polytope{D},
        prebasis::Field,
        dofs::Dof,
        face_own_dofs::Vector{Vector{Int}},
        face_own_dofs_permutations::Vector{Vector{Vector{Int}}},
        face_dofs::Vector{Vector{Int}},
        shapefuns::Field=compute_shapefuns(dofs,prebasis)) where D

  Constructs a `GenericRefFE` object with the provided data.
  """
  function GenericRefFE(
    ndofs::Int,
    polytope::Polytope{D},
    prebasis::Field,
    dofs::Dof,
    face_own_dofs::Vector{Vector{Int}},
    face_own_dofs_permutations::Vector{Vector{Vector{Int}}},
    face_dofs::Vector{Vector{Int}},
    shapefuns::Field=compute_shapefuns(dofs,prebasis)) where D

    new{D}(
      ndofs,
      polytope,
      prebasis,
      dofs,
      face_own_dofs,
      face_own_dofs_permutations,
      face_dofs,
      shapefuns)
  end
end

num_dofs(reffe::GenericRefFE) = reffe.ndofs

get_polytope(reffe::GenericRefFE) = reffe.polytope

get_prebasis(reffe::GenericRefFE) = reffe.prebasis

get_dof_basis(reffe::GenericRefFE) = reffe.dofs

get_face_own_dofs(reffe::GenericRefFE) = reffe.face_own_dofs

get_face_own_dofs_permutations(reffe::GenericRefFE) = reffe.face_own_dofs_permutations

get_face_dofs(reffe::GenericRefFE) = reffe.face_dofs

get_shapefuns(reffe::GenericRefFE) = reffe.shapefuns

