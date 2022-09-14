#ifndef EOS_EXPONENTIAL_CUTOFF_BATH_HPP
#define EOS_EXPONENTIAL_CUTOFF_BATH_HPP

#include "bath.hpp"

//#include "../discretisation.hpp"

namespace bath
{
template <typename value_type>
class exponential : public continuous_bath<value_type>, public io::registered_in_factory<abstract_bath<value_type>, exponential<value_type> >, 
                                                        public io::registered_in_factory<continuous_bath<value_type>, exponential<value_type> >
{
public:
    using base_type = continuous_bath<value_type>;
    using real_type = typename base_type::real_type;
    using complex_type = typename base_type::complex_type;
    using fourier_integ_type = typename base_type::fourier_integ_type;
    using gauss_integ_type = typename base_type::gauss_integ_type;
public:
    exponential() : base_type(), m_alpha(0), m_wc(0), m_s(1) {}
    exponential(real_type alpha, real_type wc, real_type s = 1.0) : base_type(), m_alpha(alpha), m_wc(wc), m_s(s) {}
    exponential(const IOWRAPPER::input_object& obj) : base_type()
    {
        CALL_AND_HANDLE(load(obj), "Failed to construct debye spectral density object from rapidjson value.");
    }
    ~exponential(){}

    std::shared_ptr<abstract_bath<value_type>> clone() const final{return std::make_shared<exponential<value_type>>(*this);}
    std::shared_ptr<continuous_bath<value_type>> as_continuous() const final {return std::make_shared<exponential<value_type>>(*this);}
    
    void print() final{}
    void load(const IOWRAPPER::input_object& obj) final;
        
    //functions for computing the spectral density
    real_type J(real_type w, size_t, size_t) const final
    {
        return M_PI*m_alpha/2.0 * m_wc * std::pow(std::abs(w)/m_wc, m_s) * exp(-std::abs(w)/m_wc) * (w < 0 ? -1 : 1);
    }         

    real_type S(real_type w, size_t, size_t) const final
    {
        if(this->nonzero_temperature())
        {
            if(std::abs(w) < 1e-6)
            {
                real_type beta = this->beta();  
                real_type x0 =  M_PI*m_alpha/2.0 * std::pow(w/m_wc, m_s-1) * exp(-std::abs(w)/m_wc)*2.0/beta;
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

    //functions for computing the bath correlation function
    complex_type C(real_type t, size_t, size_t) const final
    {
        ASSERT(!this->nonzero_temperature(), "Analytic Correlation function is not available for finite temperature calculations.")
        return M_PI*m_alpha/2.0 * m_wc*m_wc*std::tgamma(m_s+1)/(std::pow(complex_type(1, -m_wc*t), m_s+1));
    }      
    complex_type C(real_type t, size_t, size_t, fourier_integ_type& integ) const final
    {
        try
        {
            if(this->nonzero_temperature())
            {
                real_type Cr = integ.cosine([&](real_type w){return this->S(w,0,0);}, t, 0, this->integral_upper_bound(), trial_upper_bound());
                real_type Ci = integ.sine([&](real_type w){return this->J(w,0,0);}, t, 0, this->integral_upper_bound(), trial_upper_bound());
                return complex_type(Cr, -Ci);
            }
            else
            {
                return M_PI*m_alpha/2.0 * m_wc*m_wc*std::tgamma(m_s+1)/(std::pow(complex_type(1, -m_wc*t), m_s+1));
            }
        }
        catch(const std::exception& ex) 
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute bath correlation function.");
        }
    }      

    real_type density_of_frequencies(real_type w) const final
    {
        if(m_s < 1){return this->J(w,0,0);}
        else{return M_PI/2.0*m_alpha*std::pow(std::abs(w)/m_wc, m_s-1.0)*exp(-std::abs(w)/m_wc);}
    }
    real_type density_of_frequencies(real_type w, real_type renorm) const final
    {
        if(m_s < 1){return this->J(w,0,0)/std::sqrt(w*w+renorm*renorm);}
        else{return density_of_frequencies(w);}
    }

    std::array<real_type, 2> frequency_bounds(real_type tol, const gauss_integ_type& integ, bool use_thermofield = false) const
    {
        std::array<real_type, 2> bounds;

        if(m_s > 1)
        {
            real_type itol = 1.0-tol;
            real_type lambda = std::tgamma(m_s)*m_alpha*m_wc/2.0;
            CALL_AND_RETHROW(bounds[1] = discretisation_utilities<real_type>::find_maximum_frequency(integ, lambda*M_PI, m_wc*std::log(1.0/tol), [this](real_type w){return this->density_of_frequencies(w);}, itol));
        }
        else
        {
            bounds[1] = m_wc*std::log(1.0/tol);
        }

        if(use_thermofield && this->nonzero_temperature())
        {
            auto func = [this](real_type x){return this->density_of_frequencies(x)*exp(-this->beta()*x);};
            real_type integ_total = quad::adaptive_integrate<real_type>(func, integ, static_cast<real_type>(0.0), bounds[1], tol);
            std::cerr << integ_total << std::endl;
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

    real_type& s() {return m_s;}
    const real_type& s() const {return m_s;}

    real_type trial_upper_bound() const final{return m_wc*1e1;}
protected:
    real_type m_alpha;
    real_type m_wc;
    real_type m_s;
};
}


REGISTER_TEMPLATE_TYPE_INFO_WITH_NAME(bath::exponential, "exponential", "Generic bath with an exponential cutoff: J(\\omega) = \\frac{\\pi\\alpha }{2} \\frac{\\omega^s}{\\wc^{s-1}} e^{-\\omega/\\wc}", "wc: The cutoff frequency. \nalpha: The Kondo Parameter. \ns: The power-law dependence of the spectral density. \nbeta[T]: (O) The inverse temperature/temperature  of the bath")

namespace bath
{
template <typename value_type> 
void exponential<value_type>::load(const IOWRAPPER::input_object& obj)
{
    try
    {
        CALL_AND_HANDLE(base_type::load(obj, io::type_info<exponential<value_type> >::get_name()), "Failed to load base type variables.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "wc", m_wc), "Failed to load in cutoff frequency.");
        ASSERT(m_wc >= 0, "Invalid cutoff frequency.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "alpha", m_alpha), "Failed to load in Kondo parameter.");
        ASSERT(m_alpha >= 0, "Invalid cutoff frequency.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "s", m_s), "Failed to load in exponent.");
        ASSERT(m_s >= 0, "Invalid cutoff frequency.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to load exponential cutoff from file.");
    } 
}
        

template class exponential<double>;
}

#endif

