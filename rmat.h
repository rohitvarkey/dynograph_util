
// Adapted from the stinger project <https://github.com/stingergraph/stinger> rmat.c
#include <inttypes.h>

namespace DynoGraph {

class rmat_edge_generator {
private:
    class dxor128_generator {
    private:
        unsigned x,y,z,w;
    public:
        dxor128_generator(unsigned seed = 88675123)
            : x(123456789), y(362436069), z(521288629), w(seed) {}

        double operator() () {
            unsigned t=x^(x<<11);
            x=y; y=z; z=w; w=(w^(w>>19))^(t^(t>>8));
            return w*(1.0/4294967296.0);
        }
    };
    dxor128_generator dxor128;
    int64_t SCALE;
    double a, b, c, d;

public:
    rmat_edge_generator(int64_t nv, double a, double b, double c, double d)
    : SCALE(0), a(a), b(b), c(c), d(d)
    {
        while (nv >>= 1) { ++SCALE; }
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
            const double r = dxor128();
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
            A *= (9.5 + dxor128()) / 10;
            B *= (9.5 + dxor128()) / 10;
            C *= (9.5 + dxor128()) / 10;
            D *= (9.5 + dxor128()) / 10;
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
