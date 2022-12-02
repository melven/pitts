#include <charconv>

#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_random.hpp"

static const char* help_message = "\nNeed to pass parameters (d, n, xr, yr, iter): \n"
    "d      - dimension of tensor (order of tensortrain)\n"
    "n      - size of each dimension\n"
    "xr     - ranks of tensor x\n"
    "yr     - ranks of tensor y\n"
    "iter   - number of iterations to run\n"
    "The resulting tensors will look like:\n\n"
    "  xr/yr    xr/yr\n"
    " o -- ... o -- o\n"
    " |    ... |    |\n"
    " n    ... n    n\n"
    "   d times\n";

    // USE CASE:
    // tensor size: 20^6, 20^8, 20^10, 50^8, 100^8
    // ranks 1-150

int main(int argc, char* argv[])
{
    PITTS::initialize(&argc, &argv);

    std::size_t d=8, n=20, xr=70, yr=50, iter=10; // default values (if no cmd arguments are passed)

    switch (argc)
    {
        case 1:
            break;
        case 2:
            if (argv[1][0] == '?')
            {
                std::cout << help_message;
                return 1;
            }
            throw std::invalid_argument(help_message);
            break;
        case 6:
            std::from_chars(argv[1], argv[2], d);   // dimension
            std::from_chars(argv[2], argv[3], n);   // size of dimensions of tensors (all equal)
            std::from_chars(argv[3], argv[4], xr);  // ranks of tensortrain x (all equal)
            std::from_chars(argv[4], argv[5], yr);  // ranks of tensortrain y (all equal)
            std::from_chars(argv[5], argv[6], iter);// number of iterations
            break;
        default:
            throw std::invalid_argument(help_message);
    }
    
    PITTS::TensorTrain<double> TTx(d, n, xr), TTy(d, n, yr), TTz(d, n, yr);
    randomize(TTx);
    randomize(TTy);
    PITTS::leftNormalize(TTx, 0.0); // consider to subtract in performance results
    
    for(int i = 0; i < iter; i++)
    {
        copy(TTy, TTz); // need to be subtracted in performance results

        // benchmark here
        PITTS::internal::axpby_leftOrthogonalize(0.01, TTx, 0.9, TTz);
    }
    
    PITTS::finalize();
}