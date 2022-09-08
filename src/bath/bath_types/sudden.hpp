#ifndef EOS_SUDDEN_CUTOFF_BATH_HPP
#define EOS_SUDDEN_CUTOFF_BATH_HPP

#include "bath.hpp"
#include "../utils/factory.hpp"

namespace eos
{
namespace bath
{

template <typename value_type>
class sudden : public continuous_bath<value_type>, public registered_in_factory<abstract_bath<value_type>, sudden<value_type> >, 
                                                   public registered_in_factory<continuous_bath<value_type>, sudden<value_type> >

{
public:
    using base_type = continuous_bath<value_type>;
    using real_type = typename base_type::real_type;
    using complex_type = typename base_type::complex_type;
    using fourier_integ_type = typename base_type::fourier_integ_type;
    using gauss_integ_type = typename base_type::gauss_integ_type;
public:
    sudden() : base_type(), m_alpha(0), m_wc(0), m_s(1) {}
    sudden(real_type alpha, real_type wc, real_type s = 1.0) : base_type(), m_alpha(alpha), m_wc(wc), m_s(s) {}
    sudden(const rapidjson::Value& obj) : base_type()
    {
        CALL_AND_HANDLE(load(obj), "Failed to construct debye spectral density object from rapidjson value.");
    }
    ~sudden(){}

    std::shared_ptr<abstract_bath<value_type>> clone() const final{return std::make_shared<sudden<value_type>>(*this);}
    std::shared_ptr<continuous_bath<value_type>> as_continuous() const final {return std::make_shared<sudden<value_type>>(*this);}
    
    void print() final{}
    void load(const rapidjson::Value& obj) final;
        
    //functions for computing the spectral density

    real_type J(real_type w, size_t, size_t) const final
    {
        return (std::abs(w) < m_wc ? M_PI*m_alpha*std::tgamma(m_s+1)/2.0 * m_wc * std::pow(std::abs(w)/m_wc, m_s) : 0.0) * (w < 0 ? -1 : 1);
    }         

    real_type S(real_type w, size_t, size_t) const final
    {
        if(this->nonzero_temperature())
        {
            if(std::abs(w) < 1e-6)
            {
                real_type beta = this->beta();  
                real_type x0 = (w < m_wc ? M_PI*m_alpha*std::tgamma(m_s+1)/2.0 * std::pow(std::abs(w)/m_wc, m_s-1) : 0.0)*2.0/beta * (w < 0 ? -1 : 1);
                real_type mats = 0;
                real_type z = beta*w/2.0;
                for(size_t k=0; k<50; ++k)
                {
                    mats += 2.0*(z)/(z*z+(k+1)*(k+1)*M_PI*M_PI);
                }
                return x0 + mats*J(w,0,0);
            }
            else
            {
                return J(w,0,0)/std::tanh(w*this->beta()/2.0);
            }
        }
        return J(w,0,0);
    }         

    real_type density_of_frequencies(real_type w) const final
    {
        if(m_s < 1){return J(w,0,0);}
        else{return (std::abs(w) < m_wc ? M_PI*m_alpha*std::tgamma(m_s+1)/2.0 * std::pow(std::abs(w)/m_wc, m_s-1) : 0.0)* (w < 0 ? -1 : 1);}
    }
    real_type density_of_frequencies(real_type w, real_type renorm) const final
    {
        if(m_s < 1){return J(w,0,0)/std::sqrt(w*w+renorm*renorm);}
        else{return density_of_frequencies(w);}
    }
    std::array<real_type, 2> frequency_bounds(real_type tol, const gauss_integ_type& integ, bool use_thermofield = false) const
    {
        std::array<real_type, 2> bounds;

        bounds[1] = m_wc;

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

    real_type& s() {return m_s;}
    const real_type& s() const {return m_s;}

    real_type integral_upper_bound() const final{return m_wc;}
    real_type trial_upper_bound() const final{return m_wc;}
protected:
    real_type m_alpha;
    real_type m_wc;
    real_type m_s;
};
}


REGISTER_TEMPLATE_TYPE_INFO_WITH_NAME(bath::sudden, "sudden", "Generic bath with an sudden cutoff: J(\\omega) = \\frac{\\pi\\alpha \\Gamma(s+1)}{2} \\frac{\\omega^s}{\\wc^{s-1}} \\Theta^{wc - \\omega}", "wc: The cutoff frequency. \nalpha: The Kondo Parameter. \ns: The power-law dependence of the spectral density. \nbeta[T]: (O) The inverse temperature/temperature  of the bath")

namespace bath
{
template <typename value_type> 
void sudden<value_type>::load(const rapidjson::Value& obj)
{
    try
    {
        CALL_AND_HANDLE(base_type::load(obj, type_info<sudden<value_type> >::get_name()), "Failed to load base type variables.");

        ASSERT(obj.HasMember("wc") , "Required parameters not present in rapidjson object.");
        ASSERT(obj["wc"].IsNumber(), "Required parameters are not correctly specified.");
        m_wc = obj["wc"].GetDouble();
        ASSERT(m_wc >= 0, "Invalid cutoff frequency.");

        ASSERT(obj.HasMember("alpha") , "Required parameters not present in rapidjson object.");
        ASSERT(obj["alpha"].IsNumber(), "Required parameters are not correctly specified.");
        m_alpha = obj["alpha"].GetDouble();
        ASSERT(m_alpha >= 0, "Invalid cutoff frequency.");

        ASSERT(obj.HasMember("s") , "Required parameters not present in rapidjson object.");
        ASSERT(obj["s"].IsNumber(), "Required parameters are not correctly specified.");
        m_s = obj["s"].GetDouble();
        ASSERT(m_s >= 0, "Invalid cutoff frequency.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to load sudden cutoff from file.");
    } 
}
        

template class sudden<double>;
}
}   //namespace eos

#endif

