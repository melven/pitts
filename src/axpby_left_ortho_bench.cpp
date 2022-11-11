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

int main(int argc, char* argv[])
{
    PITTS::initialize(&argc, &argv);
    
    if (argc == 2)
        if (argv[1][0] == '?')
        {
            std::cout << help_message;
            return 1;
        }

    if(argc != 6)
        throw std::invalid_argument(help_message);

    std::size_t d=10, n=20, xr=70, yr=50, iter=1;
    std::from_chars(argv[1], argv[2], d);   // dimension
    std::from_chars(argv[2], argv[3], n);   // size of dimensions of tensors (all equal)
    std::from_chars(argv[3], argv[4], xr);  // ranks of tensortrain x (all equal)
    std::from_chars(argv[4], argv[5], yr);  // ranks of tensortrain y (all equal)
    std::from_chars(argv[5], argv[6], iter);// number of iterations
    
    PITTS::TensorTrain<double> TTx(d, n, xr), TTy(d, n, yr), TTz(d, n, yr);
    randomize(TTx);
    randomize(TTy);
    //PITTS::internal::leftNormalize(TTx, 0);
    
    for(int i = 0; i < iter; i++)
    {
        copy(TTy, TTz); // need to be subtracted in performance results

        std::cout << "\n\n" << TTz.subTensor(0).n() << "\t" << TTz.subTensor(0).r2() << "\n\n";

        // benchmark here
        PITTS::internal::axpby_leftOrthogonalize(0.01, TTx, 0.9, TTz);

        std::cout << "\n\nITERATION DONE\n\n";
    }
    
    PITTS::finalize();
}