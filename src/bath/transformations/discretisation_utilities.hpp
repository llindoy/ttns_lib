#ifndef EOS_DISCRETISATION_UTILITIES_HPP
#define EOS_DISCRETISATION_UTILITIES_HPP


namespace eos
{


template <typename T> 
class discretisation_utilities
{
public:
    template <typename Dens> 
    static inline T find_maximum_frequency(const quad::gauss::legendre<T>& leg, T l, T w0, Dens&& rho, T itol, T tol = 1e-14, size_t max_iter = 1000)   
    {
        auto df = [&](T _w){return rho(_w);};
        auto f = [&](T w){return quad::adaptive_integrate<T>(df, leg, static_cast<T>(0.0), w, tol, 0.0, true, tol);};

        return newton_optimisation(f, df, w0, l*itol, 0, tol, max_iter);
    }

    template <typename F, typename dF>
    static inline T newton_optimisation(F&& func, dF&& dfunc, T x0, T val, T xmin, T tol = 1e-14, size_t max_iter = 1000)
    {
        T x = x0;
        T delta_x = 10;
        T delta_f = 10;
    
        T df = dfunc(x);
        T f = func(x) - val;
    
        delta_f = std::abs(f);

        T step = f/df;
        if(step > 1e1){step  = 1e1;}
        if(step < -1e1){step = -1e1;}
        T xtrial = x - step;
        if(xtrial < xmin  || std::isnan(xtrial))
        {
            T xm = x;
            delta_x = std::abs(xm - x);
        }
        else
        {
            x = xtrial;
            delta_x = std::abs(f/df);
        }
        ASSERT_NUMERIC(!std::isnan(x), "Invalid frequency obtained in discretization of spectral_density.");
        size_t count = 0;
        
        while(delta_x > tol && delta_f > tol)  
        {

            T fx = func(x) - val;
            T dfx = dfunc(x); 
            delta_f = std::abs(fx);
            f = fx;
            df = dfx;
    
            step = f/df;
            if(step > 1e1){step  = 1e1;}
            if(step < -1e1){step = -1e1;}
            xtrial = x - step;
            T xm = x;
    
            if(f < 0){xmin = x;}
            if(xtrial < xmin || std::isnan(xtrial))
            {
                if(f > 0){x = (x+xmin)/2.0;}
            }
            else
            {
                if(f > 0){x = (x+xmin)/2.0;}
                x = 0.1*x + 0.9*xtrial;
            }
            delta_x = std::abs(x - xm);
            ASSERT_NUMERIC(!std::isnan(x), "Invalid frequency obtained in discretization of spectral_density.");
            ++count;
            ASSERT(count < max_iter, "Newton's method failed to converge within the allowed number of iterators.");
        }
        return x;
    }
};

}

#endif

