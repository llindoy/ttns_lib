#ifndef EOS_GAUSSIAN_CUTOFF_BATH_HPP
#define EOS_GAUSSIAN_CUTOFF_BATH_HPP

#include "bath.hpp"

namespace bath
{
template <typename value_type>
class gaussian : public continuous_bath<value_type>, public io::registered_in_factory<abstract_bath<value_type>, gaussian<value_type> >, 
                                                     public io::registered_in_factory<continuous_bath<value_type>, gaussian<value_type> >

{
public:
    using base_type = continuous_bath<value_type>;
    using real_type = typename base_type::real_type;
    using complex_type = typename base_type::complex_type;
    using fourier_integ_type = typename base_type::fourier_integ_type;
    using gauss_integ_type = typename base_type::gauss_integ_type;
public:
    gaussian() : base_type(), m_alpha(0), m_wc(0) {}
    gaussian(real_type alpha, real_type wc) : base_type(), m_alpha(alpha), m_wc(wc) {}
    gaussian(const IOWRAPPER::input_object& obj) : base_type()
    {
        CALL_AND_HANDLE(load(obj), "Failed to construct debye spectral density object from rapidjson value.");
    }
    ~gaussian(){}

    std::shared_ptr<abstract_bath<value_type>> clone() const final{return std::make_shared<gaussian<value_type>>(*this);}
    std::shared_ptr<continuous_bath<value_type>> as_continuous() const final {return std::make_shared<gaussian<value_type>>(*this);}
    
    void print() final{}
    void load(const IOWRAPPER::input_object& obj) final;
        
    //functions for computing the spectral density

    real_type J(real_type w, size_t, size_t) const final
    {
        return std::sqrt(M_PI/2)*m_alpha*w*std::exp(-w*w/(m_wc*m_wc*2));
    }         

    real_type S(real_type w, size_t, size_t) const final
    {
        if(this->nonzero_temperature())
        {
            if(std::abs(w) < 1e-6)
            {
                real_type beta = this->beta();  
                real_type x0 = std::sqrt(M_PI/2)*m_alpha*std::exp(-w*w/(m_wc*m_wc*2))*2.0/beta;
                real_type mats = 0;
                real_type z = beta*w/2.0;
                for(size_t k=0; k<50; ++k)
                {
                    mats += 2.0*(z)/(z*z+(k+1)*(k+1)*M_PI*M_PI);
                }
                return x0 + mats*this->J(w,0,0);
            }
            else
            {
                return this->J(w,0,0)/std::tanh(w*this->beta()/2.0);
            }
        }
        return this->J(w,0,0);
    }         

    real_type density_of_frequencies(real_type w) const final
    {
        return std::sqrt(M_PI/2)*m_alpha*std::exp(-w*w/(m_wc*m_wc*2));
    }
    real_type density_of_frequencies(real_type w, real_type /*renorm*/) const final
    {
        return density_of_frequencies(w);
    }

    std::array<real_type, 2> frequency_bounds(real_type tol, const gauss_integ_type& integ, bool use_thermofield = false) const
    {
        std::array<real_type, 2> bounds;

        real_type itol = 1.0-tol;
        real_type lambda = m_alpha*m_wc/2.0;
        CALL_AND_RETHROW(bounds[1] = discretisation_utilities<real_type>::find_maximum_frequency(integ, lambda*M_PI, m_wc*std::log(1.0/tol), [&](real_type w){return this->density_of_frequencies(w);}, itol));

        if(use_thermofield && this->nonzero_temperature())
        {
            auto&& func = [this](real_type x){return this->density_of_frequencies(x)*exp(-this->beta()*x);};
            real_type integ_total = quad::adaptive_integrate<real_type>(func, integ, static_cast<real_type>(0.0), bounds[1], tol);
            bounds[0] = -discretisation_utilities<real_type>::find_maximum_frequency(integ, integ_total, bounds[1]/2, func, 1-tol);
        }
        else
        {
            bounds[0] = 0;      
        }
        return bounds;
    }

    //accessors for the important
    real_type& alpha() {return m_alpha;}
    const real_type& alpha() const {return m_alpha;}

    real_type& wc() {return m_wc;}
    const real_type& wc() const {return m_wc;}

    real_type s() const {return 1;}
    real_type trial_upper_bound() const final{return m_wc*1e1;}
protected:
    real_type m_alpha;
    real_type m_wc;
};
}


REGISTER_TEMPLATE_TYPE_INFO_WITH_NAME(bath::gaussian, "gaussian", "Generic bath with a gaussian cutoff: J(\\omega) = \\sqrt\\frac{\\pi}{2} \\alpha \\omega \\exp(-\\omega^2/(2wc^2))", "wc: The cutoff frequency. \nalpha: The Kondo Parameter. \nbeta[T]: (O) The inverse temperature/temperature  of the bath")

namespace bath
{
template <typename value_type> 
void gaussian<value_type>::load(const IOWRAPPER::input_object& obj)
{
    try
    {
        CALL_AND_HANDLE(base_type::load(obj, io::type_info<gaussian<value_type> >::get_name()), "Failed to load base type variables.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "wc", m_wc), "Failed to load in cutoff frequency.");
        ASSERT(m_wc >= 0, "Invalid cutoff frequency.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "alpha", m_alpha), "Failed to load in Kondo parameter.");
        ASSERT(m_alpha >= 0, "Invalid cutoff frequency.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to load gaussian cutoff from file.");
    } 
}
        

template class gaussian<double>;
}

#endif

