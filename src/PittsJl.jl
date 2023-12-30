# Load the module and generate the functions
module PittsJl
  using CxxWrap
  @wrapmodule(() -> (return "libpitts_jl"))

  function __init__()
    @initcxx
  end

  for T = (Float32, Float64, Complex{Float32}, Complex{Float64})
    Base.unsafe_convert(to_type::Type{Ptr{T}}, t2::Tensor2View{T}) = data(t2).cpp_object
    Base.unsafe_convert(to_type::Type{Ptr{T}}, mv::MultiVector{T}) = data(mv).cpp_object

    Base.elsize(::Type{<:Tensor2View{T}}) = sizeof(T)
    Base.elsize(::Type{<:MultiVector{T}}) = sizeof(T)
  end
end

