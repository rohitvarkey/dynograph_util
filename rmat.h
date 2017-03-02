
// Adapted from the stinger project <https://github.com/stingergraph/stinger> rmat.c
#include <inttypes.h>
#include <random>
#include "prng_engine.hpp"

namespace DynoGraph {

class rmat_edge_generator {
private:
    // Generates uniformly distributed uint32_t's
    // Implements discard() in constant time
    sitmo::prng_engine rng_engine;
    // Converts values from the rng_engine into double type
    std::uniform_real_distribution<double> rng_distribution;
    double rng() { return rng_distribution(rng_engine); }

    // log2(num_vertices)
    int64_t SCALE;
    // RMAT parameters
    double a, b, c, d;

public:
    rmat_edge_generator(int64_t nv, double a, double b, double c, double d, uint32_t seed=0)
    : rng_engine(seed)
    , rng_distribution(0, 1)
    , SCALE(0), a(a), b(b), c(c), d(d)
    {
        while (nv >>= 1) { ++SCALE; }
    }

    // Skips past the next n randomly generated edges
    void discard(uint64_t n) {
        // The loop in next_edge iterates SCALE times, using 5 random numbers in each iteration
        rng_engine.discard(5 * SCALE * n);
    }

    void next_edge(int64_t *src, int64_t *dst)
    {
        double A = a;
        double B = b;
        double C = c;
        double D = d;
        int64_t i = 0, j = 0;
        int64_t bit = ((int64_t) 1) << (SCALE - 1);

        while (1) {
            const double r = rng();
            if (r > A) {                /* outside quadrant 1 */
                if (r <= A + B)           /* in quadrant 2 */
                    j |= bit;
                else if (r <= A + B + C)  /* in quadrant 3 */
                    i |= bit;
                else {                    /* in quadrant 4 */
                    j |= bit;
                    i |= bit;
                }
            }
            if (1 == bit)
                break;

            /*
              Assuming R is in (0, 1), 0.95 + 0.1 * R is in (0.95, 1.05).
              So the new probabilities are *not* the old +/- 10% but
              instead the old +/- 5%.
            */
            A *= (9.5 + rng()) / 10;
            B *= (9.5 + rng()) / 10;
            C *= (9.5 + rng()) / 10;
            D *= (9.5 + rng()) / 10;
            /* Used 5 random numbers. */

            {
                const double norm = 1.0 / (A + B + C + D);
                A *= norm;
                B *= norm;
                C *= norm;
            }
            /* So long as +/- are monotonic, ensure a+b+c+d <= 1.0 */
            D = 1.0 - (A + B + C);

            bit >>= 1;
        }
        /* Iterates SCALE times. */
        *src = i;
        *dst = j;
    }
};

} // end namespace DynoGraph
