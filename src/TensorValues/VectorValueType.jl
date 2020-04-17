###############################################################
# Types
###############################################################

"""
Type representing a first-order tensor
"""
struct VectorValue{D,T} <: MultiValue{Tuple{D},T,1,D}
    data::NTuple{D,T}
    function VectorValue{D,T}(data::NTuple{D,T}) where {D,T}
        new{D,T}(data)
    end
end

###############################################################
# Constructors (VectorValue)
###############################################################

# Empty VectorValue constructor

VectorValue()                   = VectorValue{0,Int}(NTuple{0,Int}())
VectorValue{0}()                = VectorValue{0,Int}(NTuple{0,Int}())
VectorValue{0,T}() where {T}    = VectorValue{0,T}(NTuple{0,T}())
VectorValue(data::NTuple{0})    = VectorValue{0,Int}(data)
VectorValue{0}(data::NTuple{0}) = VectorValue{0,Int}(data)

# VectorValue single NTuple argument constructor

VectorValue(data::NTuple{D,T})        where {D,T}     = VectorValue{D,T}(data)
VectorValue{D}(data::NTuple{D,T})     where {D,T}     = VectorValue{D,T}(data)
VectorValue{D,T1}(data::NTuple{D,T2}) where {D,T1,T2} = VectorValue{D,T1}(NTuple{D,T1}(data))

# VectorValue Vararg constructor
VectorValue(data::T...) where T = VectorValue{length(data),T}(data)
VectorValue{D}(data::T...)   where {D,T}   = VectorValue{D,T}(NTuple{D,T}(data))
VectorValue{D,T1}(data::T2...)   where {D,T1,T2} = VectorValue{D,T1}(NTuple{D,T1}(data))

# @fverdugo change the constructors of other types accordingly

#VectorValue(data::Real...)                  = VectorValue(NTuple{length(data)}(data))
#VectorValue{D}(data::Real...)   where {D}   = VectorValue{D}(NTuple{D}(data))
#VectorValue{D,T}(data::Real...) where {D,T} = VectorValue{D,T}(NTuple{D,T}(data))

# VectorValue single SVector, MVector and AbstractVector argument constructor



VectorValue(data::AbstractArray) = VectorValue{length(data),eltype(data)}(Tuple(data))
VectorValue{D}(data::AbstractArray) where D = VectorValue{D,eltype(data)}(NTuple{D}(data))
VectorValue{D,T}(data::AbstractArray) where {D,T} = VectorValue{D,T}(NTuple{D,T}(data))
# @fverdugo change the constructors of other types accordingly

#function VectorValue(data::
#                Union{
#                    SVector{D,T2},
#                    MVector{D,T2},
#                    AbstractArray{T2}
#                }) where {D,T1,T2}
#    PD = (@isdefined D)  ? D  : length(data)
#    VectorValue{PD,T2}(NTuple{PD,T2}(data))
#end

#function VectorValue{D}(data::
#                Union{
#                    SVector{D,T2},
#                    MVector{D,T2},
#                    AbstractArray{T2}
#                }) where {D,T1,T2}
#    VectorValue{D,T2}(NTuple{D,T2}(data))
#end
#
#function VectorValue{D,T1}(data::
#                Union{
#                    SVector{D,T2},
#                    MVector{D,T2},
#                    AbstractArray{T2}
#                }) where {D,T1,T2}
#    VectorValue{D,T1}(NTuple{D,T1}(data))
#end

###############################################################
# Conversions (VectorValue)
###############################################################

# Direct conversion
convert(::Type{<:VectorValue{D,T}},a::AbstractArray) where {D,T} = VectorValue{D,T}(a)
convert(::Type{<:VectorValue{D,T}},a::Tuple) where {D,T} = VectorValue{D,T}(a)

# Inverse conversion

convert(::Type{<:AbstractArray{T}},a::VectorValue) where T = Vector{T}(a.data)
convert(::Type{<:MVector{D,T}},a::VectorValue{D}) where D =  MVector{D,T}(a.data)
convert(::Type{<:SVector{D,T}},a::VectorValue{D}) where D =  SVector{D,T}(a.data)
convert(::Type{<:NTuple{D,T}},a::VectorValue{D}) where D =  NTuple{D,T}(a.data)

# Internal conversions
convert(::Type{<:VectorValue{D,T}},a::VectorValue{D}) where {D,T} = VectorValue{D,T}(Tuple(a))
convert(::Type{<:VectorValue{D,T}},a::VectorValue{D,T}) where {D,T} = a

#function convert(::Type{<:Union{VectorValue,VectorValue{D,T1}}}, 
#                arg::
#                    Union{
#                        NTuple{D,T2},
#                        SVector{D,T2},
#                        MVector{D,T2},
#                        AbstractArray{T2}
#                    }) where {D,T1,T2}
#    PT = (@isdefined T1) ? T1 : T2
#    PD = (@isdefined D)  ? D  : length(arg)
#    VectorValue{PD,PT}(NTuple{PD,PT}(arg))
#end
#
#function convert(::Type{<:Union{NTuple,NTuple{D,T1}}}, arg::VectorValue{D,T2}) where {D,T1,T2}
#    PT = (@isdefined T1) ? T1 : T2
#    NTuple{D,PT}(arg.data)
#end
#
#function convert(::Type{<:Union{SVector,SVector{D,T1}}}, arg::VectorValue{D,T2}) where {D,T1,T2}
#    PT = (@isdefined T1) ? T1 : T2
#    SVector{D,PT}(arg.data)
#end
#
#function convert(::Type{<:Union{MVector,MVector{D,T1}}}, arg::VectorValue{D,T2}) where {D,T1,T2}
#    PT = (@isdefined T1) ? T1 : T2
#    MVector{D,PT}(arg.data)
#end
#
#function convert(::Type{<:Union{VectorValue,VectorValue{D,T1}}}, arg::VectorValue{D,T2}) where {D,T1,T2}
#    PT = (@isdefined T1) ? T1 : T2
#    PT == T2 ? arg : convert(VectorValue{D,PT}, arg.data)
#end

###############################################################
# Other constructors and conversions (VectorValue)
###############################################################

zero(::Type{<:VectorValue{D,T}}) where {D,T} = VectorValue(tfill(zero(T),Val{D}()))
#zero(::Type{<:VectorValue{D,T}}) where {D,T} = VectorValue{D,T}(NTuple{D,T}(zeros(T,D)))
zero(::VectorValue{D,T}) where {D,T} = zero(VectorValue{D,T})

# @fverdugo solo tiene sentido para TensorValue i SymTensorValueType i SymFourthOrderTensorValue
#one(::Type{<:VectorValue{D,T}}) where {D,T} = VectorValue(tfill(one(T),Val{D}()))
##one(::Type{<:VectorValue{D,T}}) where {D,T} = VectorValue{D,T}(NTuple{D,T}(ones(T,D)))
#one(::VectorValue{D,T}) where {D,T} = one(VectorValue{D,T})

mutable(::Type{VectorValue{D,T}}) where {D,T} = MVector{D,T}
mutable(::VectorValue{D,T}) where {D,T} = mutable(VectorValue{D,T})

change_eltype(::Type{VectorValue{D}},::Type{T}) where {D,T} = VectorValue{D,T}
change_eltype(::Type{VectorValue{D,T1}},::Type{T2}) where {D,T1,T2} = VectorValue{D,T2}
change_eltype(::VectorValue{D,T1},::Type{T2}) where {D,T1,T2} = change_eltype(VectorValue{D,T1},T2)

#SVector(arg::VectorValue{D,T}) where {D,T} = SVector{D,T}(arg.data)
#SArray(arg::VectorValue{D,T}) where {D,T} = SVector(arg)
get_array(arg::VectorValue{D,T}) where {D,T} = convert(SVector{D,T},arg)
#get_array(arg::T where {T<:VectorValue}) = convert(SVector,arg)

###############################################################
# Introspection (VectorValue)
###############################################################

eltype(::Type{<:VectorValue{D,T}}) where {D,T} = T
eltype(arg::VectorValue{D,T}) where {D,T} = eltype(VectorValue{D,T})

size(::Type{VectorValue{D}}) where {D} = (D,)
size(::Type{VectorValue{D,T}}) where {D,T} = (D,)
size(::VectorValue{D,T}) where {D,T}  = size(VectorValue{D,T})

length(::Type{VectorValue{D}}) where {D} = D
length(::Type{VectorValue{D,T}}) where {D,T} = D
length(::VectorValue{D,T}) where {D,T} = length(VectorValue{D,T})

# num_components
n_components(::Type{VectorValue{D}}) where {D} = length(VectorValue{D})
n_components(::Type{VectorValue{D,T}}) where {D,T} = length(VectorValue{D,T})
n_components(::VectorValue{D,T}) where {D,T} = n_components(VectorValue{D,T})

