#ifndef EOS_CHAIN_BATH_HPP
#define EOS_CHAIN_BATH_HPP

#include "bath.hpp"
#include "discrete_bath.hpp"
#include <orthopol.hpp>

namespace bath
{

template <typename value_type>
class chain_mapped_bath : public abstract_bath<value_type>, public io::registered_in_factory<abstract_bath<value_type>, chain_mapped_bath<value_type> >
{
public:
    using base_type = abstract_bath<value_type>;
    using real_type = typename base_type::real_type;
    using complex_type = typename base_type::complex_type;
    using fourier_integ_type = typename base_type::fourier_integ_type;
public:
    chain_mapped_bath() : base_type(), m_weights_constructed(false) {}
    chain_mapped_bath(size_t N) : base_type(), m_eps(N), m_t(N-1), m_weights_constructed(false) {}
    chain_mapped_bath(const linalg::vector<real_type>& eps, const real_type& kappa, const linalg::vector<real_type>& t) : base_type(), m_kappa(kappa), m_eps(eps), m_t(t), m_weights_constructed(false) {ASSERT(eps.size() == t.size()+1, "Failed to construct chain mapped bath.  Incorrect array sizes.");}
    chain_mapped_bath(const IOWRAPPER::input_object& obj) : base_type()
    {
        CALL_AND_HANDLE(load(obj), "Failed to construct debye spectral density object from rapidjson value.");
    }
    ~chain_mapped_bath(){}

    std::shared_ptr<abstract_bath<value_type>> clone() const final{return std::make_shared<chain_mapped_bath<value_type>>(*this);}
    std::shared_ptr<chain_mapped_bath<value_type>> as_chain_mapped() const final{return std::make_shared<chain_mapped_bath<value_type>>(*this);}

    void resize(size_t N){m_eps.resize(N); m_t.resize(N-1);}
    void print() final{}
    void load(const IOWRAPPER::input_object& obj) final;
        
    bool is_chain_mapped() const final{return true;}

    //functions for computing the spectral density
    real_type J(real_type, size_t, size_t) const final
    {
        RAISE_EXCEPTION("Spectral density is not available for chain mapped baths. Convert to a discrete bath to see its spectral density.");
    }         

    real_type S(real_type, size_t, size_t) const final
    {
        RAISE_EXCEPTION("Spectral density is not available for chain mapped baths. Convert to a discrete bath to see its spectral density.");
    }         

    //functions for computing the bath correlation function
    complex_type C(real_type t, size_t mi, size_t mj, fourier_integ_type& /*integ*/) const final
    {
        ASSERT(m_weights_constructed, "Cannot compute the correlation function unless the orthonormal polynomial object has been fully constructed.");
        ASSERT(mi < base_type::m_nterms && mj < base_type::m_nterms, "Index out of bounds.");
        real_type Cr = 0.0;
        real_type Ci = 0.0;
        for(size_t i = 0; i < m_poly.npoints(); ++i)
        {
            real_type w = m_poly.node(i);
            real_type c = m_poly.weight(i);
            Cr += c*std::cos(w*t);
            Ci += c*std::sin(w*t);
        }
        return complex_type(Cr, -Ci);
    }      

    complex_type eta(real_type t, size_t mi, size_t mj, fourier_integ_type& /*integ*/) const final
    {
        ASSERT(m_weights_constructed, "Cannot compute the correlation function unless the orthonormal polynomial object has been fully constructed.");
        ASSERT(mi < base_type::m_nterms && mj < base_type::m_nterms, "Index out of bounds.");
        real_type etar = 0.0;
        real_type etai = 0.0;
        for(size_t i = 0; i < m_poly.npoints(); ++i)
        {
            real_type w = m_poly.node(i);
            real_type c = m_poly.weight(i);
            etar += c*abstract_bath<value_type>::eta_real_factor(w, t);
            etai += c*abstract_bath<value_type>::eta_imag_factor(w, t);
        }
        return complex_type(etar, etai);
    }

    //accessors for the important
    const real_type& t(size_t i) const{ASSERT(i < m_t.size()+1, "Index out of bounds.");  if(i == 0){return m_kappa;} else{return m_t[i-1];}}
    real_type& t(size_t i){ASSERT(i < m_t.size()+1, "Index out of bounds.");  m_weights_constructed = false; if(i == 0){return m_kappa;} else{return m_t[i-1];}  }

    const real_type& eps(size_t i) const{ASSERT(i < m_eps.size(), "Index out of bounds.");  return m_eps[i];}
    real_type& eps(size_t i){ASSERT(i < m_eps.size(), "Index out of bounds.");  m_weights_constructed = false; return m_eps[i];}

    const real_type& kappa() const{return m_kappa;}
    real_type& kappa(){m_weights_constructed = false; return m_kappa;}

    void setup_orthopols()
    {
        CALL_AND_HANDLE(m_poly.resize(m_eps.size()), "Failed to resize orthonormal polynomial object.");
        for(size_t i = 0; i < m_eps.size(); ++i){m_poly.set_alpha(i, m_eps[i]);}
        for(size_t i = 0; i < m_t.size(); ++i){m_poly.set_beta(i, m_t[i]*m_t[i]);}
        CALL_AND_HANDLE(m_poly.compute_nodes_and_weights(), "Failed to compute nodes and weights of the orthonormal polynomial.");
        m_weights_constructed = true;
    }

    real_type trial_upper_bound() const final{return 0.0;}

    const orthopol<real_type>& poly() const{return m_poly;}
protected:
    real_type m_kappa;
    linalg::vector<real_type> m_eps;
    linalg::vector<real_type> m_t;
    bool m_weights_constructed;
    real_type m_wmin;

    orthopol<real_type> m_poly;
};
}


REGISTER_TEMPLATE_TYPE_INFO_WITH_NAME(bath::chain_mapped_bath, "chain_mapped", "A representation of the spectral density following a chain mapping procedure.  Specified in terms of the three term recurrence relation for the orthonormal polynomials and the coupling constant arising from integration of the spectral density.", "Either:\nkappa: The system bath coupling constant. \neps: The bath site energys.\nt: The bath hoping constants.\n\ncontinuous: A continuous spectral density object.\ndiscretisation: The parameters specifying the discretisation strategy to form the discrete representation of the bath (Only quadrature based discretisation is supported for the chain mapped bath. See discretisation for more details).\nOr:\ndiscrete: A discrete bath object (see discrete bath for more details). This transformation is performed using the Lanczos algorithm.")


namespace bath
{
template <typename value_type> 
void chain_mapped_bath<value_type>::load(const IOWRAPPER::input_object& obj)
{
    try
    {
        CALL_AND_HANDLE(base_type::load(obj, io::type_info<chain_mapped_bath<value_type> >::get_name()), "Failed to load base type variables.");

        if(IOWRAPPER::has_member(obj, "eps") && IOWRAPPER::has_member(obj, "t") && IOWRAPPER::has_member(obj, "kappa"))
        {
            linalg::vector<real_type> eps;
            linalg::vector<real_type> t;
            CALL_AND_HANDLE(IOWRAPPER::load<decltype(eps)>(obj, "eps", ps), "Failed to load in site energy array from rapidjson object.");
            CALL_AND_HANDLE(IOWRAPPER::load<decltype(t)>(obj, "t", t), "Failed to load bath-bath coupling constant array from rapidjson object.");
            CALL_AND_HANDLE(IOWRAPPER::load<decltype(m_kappa)>(obj, "kappa", m_kappa), "Failed to read in system bath coupling constant.");

            ASSERT(eps.size() == t.size()+1, "Invalid spectral_density parameters, the number of coupling constants and frequencies are inconsistent.");
            
            CALL_AND_HANDLE(m_t = t, "Failed to copy the coupling constants to internal storage.");
            CALL_AND_HANDLE(m_eps = eps, "Failed to copy the site energies to internal storage.");

            m_weights_constructed = false;

            base_type::m_nterms = 1;


        }
        else if(obj.HasMember("continuous") && obj.HasMember("discretisation"))
        {
        }
        else if(obj.HasMember("discrete"))
        {
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to load chain_mapped_bath bath from file.");
    } 
}
        

template class chain_mapped_bath<double>;

}

#endif

