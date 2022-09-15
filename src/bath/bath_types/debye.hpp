#ifndef EOS_DEBYE_CUTOFF_BATH_HPP
#define EOS_DEBYE_CUTOFF_BATH_HPP

#include "bath.hpp"

namespace bath
{
template <typename value_type>
class debye : public continuous_bath<value_type>, public io::registered_in_factory<abstract_bath<value_type>, debye<value_type> >, 
                                                  public io::registered_in_factory<continuous_bath<value_type>, debye<value_type> >
{
public:
    using base_type = continuous_bath<value_type>;
    using real_type = typename base_type::real_type;
    using complex_type = typename base_type::complex_type;
    using fourier_integ_type = typename base_type::fourier_integ_type;
    using gauss_integ_type = typename base_type::gauss_integ_type;
public:
    debye() : base_type(), m_alpha(0), m_wc(0) {}
    debye(real_type alpha, real_type wc) : base_type(), m_alpha(alpha), m_wc(wc) {}
    debye(const IOWRAPPER::input_object& obj) : base_type()
    {
        CALL_AND_HANDLE(load(obj), "Failed to construct debye spectral density object from rapidjson value.");
    }
    ~debye(){}

    std::shared_ptr<abstract_bath<value_type>> clone() const final{return std::make_shared<debye<value_type>>(*this);}
    std::shared_ptr<continuous_bath<value_type>> as_continuous() const final {return std::make_shared<debye<value_type>>(*this);}
    
    void print() final{}
    void load(const IOWRAPPER::input_object& obj) final;
        
    //functions for computing the spectral density

    real_type J(real_type w, size_t, size_t) const final
    {
        return m_alpha*m_wc*m_wc*w/(w*w+m_wc*m_wc);
    }         

    real_type S(real_type w, size_t, size_t ) const final
    {
        if(this->nonzero_temperature())
        {
            if(std::abs(w) < 1e-6)
            {
                real_type beta = this->beta();  
                real_type x0 = m_alpha*m_wc*m_wc/(w*w+m_wc*m_wc)*2.0/beta;
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



    real_type density_of_frequencies(real_type w) const final{return m_alpha*m_wc*m_wc/(2.0*(w*w+m_wc*m_wc));}
    real_type density_of_frequencies(real_type w, real_type /*renorm*/) const final{return density_of_frequencies(w);}

    std::array<real_type, 2> frequency_bounds(real_type tol, const gauss_integ_type& integ, bool use_thermofield = false) const
    {
        std::array<real_type, 2> bounds;
        if(use_thermofield && this->nonzero_temperature())
        {
            bounds[1] = m_wc*std::tan((1.0-tol)*M_PI/2.0);
            auto func = [this](real_type x){return this->density_of_frequencies(x)*exp(-this->beta()*x);};
            real_type integ_total = quad::adaptive_integrate<real_type>(func, integ, static_cast<real_type>(0.0), bounds[1], tol);
            bounds[0] = -discretisation_utilities<real_type>::find_maximum_frequency(integ, integ_total, bounds[1]/2, func, 1-tol);
        }
        else
        {
            bounds[0] = 0;      
            bounds[1] = m_wc*std::tan((1.0-tol)*M_PI/2.0);
        }
        return bounds;
    }

    //accessors for the important
    real_type& alpha() {return m_alpha;}
    const real_type& alpha() const {return m_alpha;}

    real_type& wc() {return m_wc;}
    const real_type& wc() const {return m_wc;}

    real_type s() const {return 1;}
    real_type trial_upper_bound() const final{return m_wc*1e2;}
protected:
    real_type m_alpha;
    real_type m_wc;
};
}


REGISTER_TEMPLATE_TYPE_INFO_WITH_NAME(bath::debye, "debye", "Debye spectral density: J(\\omega) = \\alpha\\omega_c^2 \\frac{\\omega}{\\omega^2 + \\omega_c^2} or equivalently J(\\omega) = 2\\lambda\\omega_c \\frac{\\omega}{\\omega^2 + \\omega_c^2}", "wc: The cutoff frequency. \nalpha/lambda: The Kondo Parameter/bath reorganisation energy. \nbeta[T]: (O) The inverse temperature/temperature  of the bath")

namespace bath
{
template <typename value_type> 
void debye<value_type>::load(const IOWRAPPER::input_object& obj)
{
    try
    {
        CALL_AND_HANDLE(base_type::load(obj, io::type_info<debye<value_type> >::get_name()), "Failed to load base type variables.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "wc", m_wc), "Failed to load in cutoff frequency.");
        ASSERT(m_wc >= 0, "Invalid cutoff frequency.");

        bool loaded_alpha = false;
        CALL_AND_HANDLE(loaded_alpha = IOWRAPPER::load_optional<real_type>(obj, "alpha", m_alpha), "Failed to load in bath friction.");
        if(!loaded_alpha)
        {
            real_type lambda;
            bool loaded_lambda;
            CALL_AND_HANDLE(loaded_lambda = IOWRAPPER::load_optional<real_type>(obj, "lambda", lambda), "Failed to load in bath friction.");
            ASSERT(loaded_lambda, "Failed to read in the bath coupling strength.");

            m_alpha = real_type(2)*lambda/m_wc;
        }
        ASSERT(m_alpha >= 0, "Invalid cutoff frequency.");

        
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to load debye cutoff from file.");
    } 
}
        

template class debye<double>;
}

#endif

