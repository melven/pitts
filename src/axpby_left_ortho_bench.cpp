#include <charconv>

#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_random.hpp"

int main(int argc, char* argv[])
{
    PITTS::initialize(&argc, &argv);
    
    if(argc != 6)
        throw std::invalid_argument("Requires exactly 4 arguments (d, n, xr, yr, iter)");

    std::size_t d, n, xr, yr, iter;
    std::from_chars(argv[1], argv[2], d);   // dimension
    std::from_chars(argv[2], argv[3], n);   // size of dimensions of tensors (all equal)
    std::from_chars(argv[3], argv[4], xr);  // ranks of tensortrain x (all equal)
    std::from_chars(argv[4], argv[5], yr);  // ranks of tensortrain y (all equal)
    std::from_chars(argv[5], argv[6], iter);// number of iterations
    
    using Type = double;
    PITTS::TensorTrain<Type> TTx(d, n, xr), TTy(d, n, yr);
    
    for(int i = 0; i < iter; i++)
    {
        randomize(TTx);
        randomize(TTy);
        PITTS::internal::leftNormalize_range<Type>(TTx, 0, d-1, 0);
        PITTS::internal::axpby_leftOrthogonalize(0.01, TTx, 0.9, TTy);
    }
    
    PITTS::finalize();
}