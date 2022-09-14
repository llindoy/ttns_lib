#ifndef EOS_BROWNIAN_OSCILLATOR_BATH_HPP
#define EOS_BROWNIAN_OSCILLATOR_BATH_HPP

#include "bath.hpp"

namespace bath
{
template <typename value_type>
class brownian_oscillator : public abstract_bath<value_type>, public io::registered_in_factory<abstract_bath<value_type>, brownian_oscillator<value_type> >
{
public:
    using base_type = abstract_bath<value_type>;
    using real_type = typename base_type::real_type;
    using complex_type = typename base_type::complex_type;
    using fourier_integ_type = typename base_type::fourier_integ_type;
public:
    brownian_oscillator() : base_type(), m_alpha(0), m_wc(0) {}
    brownian_oscillator(real_type alpha, real_type wc) : base_type(), m_alpha(alpha), m_wc(wc) {}
    brownian_oscillator(const IOWRAPPER::input_object& obj) : base_type()
    {
        CALL_AND_HANDLE(load(obj), "Failed to construct brownian_oscillator spectral density object from rapidjson value.");
    }
    ~brownian_oscillator(){}

    std::shared_ptr<abstract_bath<value_type>> clone() const final{return std::make_shared<brownian_oscillator<value_type>>(*this);}
    
    void print() final{}
    void load(const IOWRAPPER::input_object& obj) final;
        
    //functions for computing the spectral density

    real_type J(real_type w, size_t , size_t ) const final
    {
    }         

    real_type S(real_type w, size_t , size_t ) const final
    {

    }         

    real_type density_of_frequencies(real_type w) const final
    {
        //if(m_s < 1){return J(w,0,0);}
        //else{return std::sqrt(M_PI/2)*m_alpha*std::exp(-w*w/(m_wc*m_wc*2));}
    }
    real_type density_of_frequencies(real_type w, real_type /*renorm*/) const final
    {
        //return density_of_frequencies(w);
    }
    real_type frequency_upper_bound(size_t N) const final
    {
        //real_type itol = N/(N+1.0);
        //real_type lambda = m_alpha*m_wc/2.0;
        //return density_discretisation<value_type>::find_maximum_frequency(lambda, m_wc*std::log(N+1.0), [&](real_type w){return this->density_of_frequencies(w);}, itol);
    }



    //accessors for the important
    real_type& alpha() {return m_alpha;}
    const real_type& alpha() const {return m_alpha;}

    real_type& Omega() {return m_Omega;}
    const real_type& Omega() const {return m_Omega;}

    real_type& gamma() {return m_gamma;}
    const real_type& gamma() const {return m_gamma;}

    real_type s() const {return 1;}
    real_type trial_upper_bound() const final{return m_wc*1e2;}
protected:
    real_type m_alpha;
    real_type m_Omega;
    real_type m_gamma;
};
}


REGISTER_TEMPLATE_TYPE_INFO_WITH_NAME(bath::brownian_oscillator, "brownian_oscillator", "Debye spectral density: J(\\omega) = \\alpha\\omega_c^2 \\frac{\\omega}{\\omega^2 + \\omega_c^2}", "wc: The cutoff frequency. \nalpha: The Kondo Parameter. \nbeta[T]: (O) The inverse temperature/temperature  of the bath")

namespace bath
{
template <typename value_type> 
void brownian_oscillator<value_type>::load(const IOWRAPPER::input_object& obj)
{
    try
    {
        CALL_AND_HANDLE(base_type::load(obj, io::type_info<brownian_oscillator<value_type> >::get_name()), "Failed to load base type variables.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "gamma", m_gamma), "Failed to load reaction coordinate friction.");
        ASSERT(m_gamma >= 0, "Invalid cutoff frequency.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "omega", m_Omega), "Failed to load reaction coordinate frequency.");
        ASSERT(m_Omega >= 0, "Invalid cutoff frequency.");

        CALL_AND_HANDLE(IOWRAPPER::load<real_type>(obj, "alpha", m_alpha), "Failed to load reaction coordinate frequency.");
        ASSERT(m_alpha >= 0, "Invalid cutoff frequency.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to load brownian_oscillator cutoff from file.");
    } 
}
        

template class brownian_oscillator<double>;
}

#endif

