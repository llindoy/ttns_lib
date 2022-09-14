#ifndef EOS_DENSITY_DISCRETISATION_HPP
#define EOS_DENSITY_DISCRETISATION_HPP

#include "discretisation.hpp"
#include <quadrature/adaptive_integrate.hpp>
#include <quadrature/gaussian_quadrature/gauss_legendre_quadrature.hpp>



template <typename T>
class density_discretisation //: public abstract_discretisation<T>, public registered_in_factory<abstract_discretisation<T>, density_discretisation<T> >
{
public:
    using real_type = T;
    using complex_type = linalg::complex<real_type>;
public:

    template <typename Dens> 
    static inline real_type generate_frequencies(std::vector<real_type>& wk, Dens&& rho, real_type wmax, real_type tol = 1e-14, size_t max_iter = 1000)
    {
        quad::gauss::legendre<real_type> leg(100);

        //determine the normalisation factor for the density so that \int_0^\omega_{max} \rho(\omega) \mathrm{d}\omega = N
        real_type c = quad::adaptive_integrate<real_type>([&](real_type _w){return rho(_w);}, leg, static_cast<real_type>(0.0), wmax, 1e-14, 0.0, true, 1e-14);
        size_t N = wk.size();
        real_type a1 = N/c;
        
        real_type w = wmax;
        for(size_t i=0; i < N; ++i)
        {
            size_t j = N-i;
            size_t ind = j-1;
        
            auto df = [&](real_type _w){return rho(_w)*a1;};
            auto f = [&](real_type _w){return quad::adaptive_integrate<real_type>(df, leg, static_cast<real_type>(0.0), _w, tol, 0.0, true, tol);};
            w = newton_optimisation(f, df, w, j, 0, tol, max_iter);
            wk[ind] = w;
        }
        return a1;
    }



protected:


    
};

#endif
