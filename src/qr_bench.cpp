#include "pitts_mkl.hpp"
#include "pitts_common.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_split.hpp"

static const char* help_message = "\nOnly allowed parameters are f, h, l, ?: \n"
    "f      - take QR of full-rank matrix\n"
    "h      - take QR of high-rank matrix\n"
    "l      - take QR of low-rank matrix\n"
    "?      - display this help message and exit\n";


int main(int argc, char* argv[])
{
    PITTS::initialize(&argc, &argv);

    std::vector<std::size_t> rows = {16,70,264,1032,2056,4104}, cols = {16,70,264,1032,2056,4104};
    float rank_multiplier = 0.75;
    std::size_t rnk;

    switch (argc)
    {
    case 1:
        break;
    case 2:
        if      (argv[1][0] == 'f')
            rank_multiplier = 1.0;
        else if (argv[1][0] == 'h')
            rank_multiplier = 0.66;
        else if (argv[1][0] == 'l')
            rank_multiplier = 0.33;
        else if (argv[1][0] == '?')
        {
            std::cout << help_message;
            return 1;
        }
        break;
    default:
        throw std::invalid_argument(help_message);
    }
    
    for (std::size_t row : rows)
    {
        for (std::size_t col : cols)
        {
            rnk = std::min(row, col) * rank_multiplier;
            PITTS::Tensor2<double> A(row, col), B(row, rnk), C(rnk, col);
            randomize(B);
            randomize(C);
            EigenMap(A) = ConstEigenMap(B) * ConstEigenMap(C);

            for (int i = 0; i < 10; i++)
            {
                auto [Q,R] = PITTS::internal::normalize_qb(A, true, 0.0, row+col);
            }
        }
    }
    
    PITTS::finalize();
}
