# Load the module and generate the functions
module PittsJl
  using CxxWrap
  @wrapmodule(() -> (return "libpitts_jl"))

  function __init__()
    @initcxx
  end
end

