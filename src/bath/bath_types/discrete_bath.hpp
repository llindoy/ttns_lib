#ifndef EOS_DISCRETE_BATH_HPP
#define EOS_DISCRETE_BATH_HPP

#include "bath.hpp"
#include "chain_bath.hpp"

namespace bath
{

template <typename value_type>
class discrete_bath : public abstract_bath<value_type>, public io::registered_in_factory<abstract_bath<value_type>, discrete_bath<value_type> >
{
public:
    using base_type = abstract_bath<value_type>;
    using real_type = typename base_type::real_type;
    using complex_type = typename base_type::complex_type;
    using fourier_integ_type = typename base_type::fourier_integ_type;
public:
    discrete_bath() : base_type(), m_jeps(1e-6) {}
    discrete_bath(size_t N, size_t nterms) : base_type(nterms), m_w(N), m_g(N), m_jeps(1e-6) 
    {
        for(size_t i = 0; i < m_g.size(); ++i){m_g(i).resize(nterms);}
    }
    discrete_bath(const linalg::vector<real_type>& w, const linalg::vector<linalg::vector<real_type>>& g) : base_type(), m_w(w), m_g(g), m_jeps(1e-6) {}
    discrete_bath(const IOWRAPPER::input_object& obj) : base_type(), m_jeps(1e-6)
    {
        CALL_AND_HANDLE(load(obj), "Failed to construct debye spectral density object from rapidjson value.");
    }
    ~discrete_bath(){}

    std::shared_ptr<abstract_bath<value_type>> clone() const final{return std::make_shared<discrete_bath<value_type>>(*this);}
    std::shared_ptr<discrete_bath<value_type>> as_discrete() const final{return std::make_shared<discrete_bath<value_type>>(*this);}

    void resize(size_t N){m_w.resize(N); m_g.resize(N);}
    void print() final{}
    void load(const IOWRAPPER::input_object& obj) final;
        
    bool is_discrete() const final{return true;}

    //functions for computing the spectral density
    real_type J(real_type w, size_t mi, size_t mj) const final
    {
        ASSERT(mi < base_type::m_nterms && mj < base_type::m_nterms, "Index out of bounds.");
        real_type jw = 0.0;
        for(size_t i = 0; i < m_w.size(); ++i)
        {
            jw += M_PI*m_g[i][mi]*m_g[i][mj] * 1.0/(2.0*std::sqrt(m_jeps*M_PI)) * std::exp(-(w-m_w[i])*(w-m_w[i])/(2.0*m_jeps));
        }
        return jw;
    }         

    real_type S(real_type w, size_t mi, size_t mj) const final
    {
        ASSERT(mi < base_type::m_nterms && mj < base_type::m_nterms, "Index out of bounds.");
        real_type jw = 0.0;
        for(size_t i = 0; i < m_w.size(); ++i)
        {
            jw += M_PI*m_g[i][mi]*m_g[i][mj] * 1.0/(2.0*std::sqrt(m_jeps*M_PI)) * std::exp(-(w-m_w[i])*(w-m_w[i])/(2.0*m_jeps))/std::tanh(this->beta()*w/2.0);
        }
        return jw;
    }         

    //functions for computing the bath correlation function
    complex_type C(real_type t, size_t mi, size_t mj, fourier_integ_type& /*integ*/) const final
    {
        ASSERT(mi < base_type::m_nterms && mj < base_type::m_nterms, "Index out of bounds.");
        if(this->nonzero_temperature())
        {
            real_type Cr = 0.0;
            real_type Ci = 0.0;
            for(size_t i = 0; i < m_w.size(); ++i)
            {
                Cr += m_g[i][mi]*m_g[i][mj]*std::cos(m_w[i]*t)/std::tanh(this->beta()*m_w[i]/2.0);
                Ci += m_g[i][mi]*m_g[i][mj]*std::sin(m_w[i]*t);
            }
            return complex_type(Cr, -Ci);
        }
        else
        {
            complex_type res(0, 0);
            for(size_t i = 0; i < m_w.size(); ++i)
            {
                res += m_g[i][mi]*m_g[i][mj]*std::exp(complex_type(0, m_w[i]*t));
            }
            return M_PI*res;
        }
    }      

    complex_type eta(real_type t, size_t mi, size_t mj, fourier_integ_type& /*integ*/) const final
    {
        ASSERT(mi < base_type::m_nterms && mj < base_type::m_nterms, "Index out of bounds.");
        real_type etar = 0.0;
        real_type etai = 0.0;

        for(size_t i = 0; i < m_w.size(); ++i)
        {
            etai += m_g[i][mi]*m_g[i][mj]*abstract_bath<value_type>::eta_imag_factor(m_w[i], t);
        }

        if(this->nonzero_temperature())
        {
            for(size_t i = 0; i < m_w.size(); ++i)
            {
                etar += m_g[i][mi]*m_g[i][mj]*abstract_bath<value_type>::eta_real_factor(m_w[i], t)/std::tanh(this->beta()*m_w[i]/2.0);
            }
        }
        else
        {
            for(size_t i = 0; i < m_w.size(); ++i)
            {
                etar += m_g[i][mi]*m_g[i][mj]*abstract_bath<value_type>::eta_real_factor(m_w[i], t);
            }
        }
        return M_PI*complex_type(etar, etai);
    }

    //accessors for the important


    const linalg::vector<real_type>& w() const{return m_w;}
    linalg::vector<real_type>& w() {return m_w;}

    const real_type& w(size_t i) const{ASSERT(i < m_w.size(), "Index out of bounds.");  return m_w[i];}
    real_type& w(size_t i){ASSERT(i < m_w.size(), "Index out of bounds.");  return m_w[i];}

    const linalg::vector<linalg::vector<real_type>>& g() const{return m_g;}
    linalg::vector<linalg::vector<real_type>>& g() {return m_g;}

    const linalg::vector<real_type>& g(size_t i) const{ASSERT(i < m_g.shape(0), "Index out of bounds.");  return m_g[i];}
    linalg::vector<real_type>& g(size_t i) {ASSERT(i < m_g.shape(0), "Index out of bounds.");  return m_g[i];}

    const real_type& g(size_t i, size_t j) const{ASSERT(i < m_g.shape(0) && j < m_g(0).shape(0), "Index out of bounds.");  return m_g[i][j];}
    real_type& g(size_t i, size_t j){ASSERT(i < m_g.shape(0) && j < m_g(0).shape(0), "Index out of bounds.");  return m_g[i][j];}


    real_type trial_upper_bound() const final{return 0.0;}
protected:
    linalg::vector<real_type> m_w;
    linalg::vector<linalg::vector<real_type>> m_g;
    real_type m_jeps;
};
}


REGISTER_TEMPLATE_TYPE_INFO_WITH_NAME(bath::discrete_bath, "discrete", "The spectral density for a discrete set of harmonic oscillators: J(\\omega) = \\pi\\sum_i^N \\g_i^2 \\delta(\\omega - \\omega_i)", "Either:\nw: A real array of the frequencies of each mode in the spectral density. \ng: An array of (real arrays or real numbers) specifying the system coupling constants associate with each mode in the spectral_density.\nOr:\ncontinuous: A continuous spectral density object.\ndiscretisation: The parameters specifying the discretisation strategy to form the discrete representation of the bath (see discretisation for more details).\nOr:\nchain_mapped: A chain mapped bath object (see chain mapped bath for more details).")


namespace bath
{
template <typename value_type> 
void discrete_bath<value_type>::load(const IOWRAPPER::input_object& obj)
{
    try
    {
        CALL_AND_HANDLE(base_type::load(obj, io::type_info<discrete_bath<value_type> >::get_name()), "Failed to load base type variables.");

        //need to figure out how to best handle finite temperature cases here.
        if(IOWRAPPER::has_member(obj, "w") && IOWRAPPER::has_member(obj, "g"))
        {
            linalg::vector<real_type> w;
            linalg::vector<linalg::vector<real_type>> g;

            CALL_AND_HANDLE(IOWRAPPER::load<decltype(w)>(obj, "w", w), "Failed to load frequency array from rapidjson object.");
            CALL_AND_HANDLE(IOWRAPPER::load<decltype(g)>(obj, "g", g), "Failed to load coupling constants array from rapidjson object.");

            ASSERT(w.size() == g.size(), "Invalid spectral_density parameters, the number of coupling constants and frequencies are inconsistent.");
            
            m_w = w;

            size_t nterms = 0;
            for(size_t i = 0; i < g.size(); ++i)
            {
                if(nterms == 0){nterms = g(i).size();}
                ASSERT(g(i).size() == nterms, "The coupling constants are not all the same size.");
            }           

            base_type::m_nterms = nterms;
            m_g.resize(g.size());
            for(size_t i = 0; i < g.size(); ++i){m_g(i) = g(i);}
        
            CALL_AND_HANDLE(IOWRAPPER::load_optional<bool>(obj, "containsthermal", base_type::m_nonzero_temperature), "Failed to read in whether or not the discrete bath represents a thermofield mapped bath.");
        }
        else if(IOWRAPPER::has_member(obj, "discretisation") && IOWRAPPER::has_member(obj, "continuous"))
        {
            try
            {
                //ASSERT(obj["continuous"].IsObject() && obj["discretisation"].IsObject(), "Invalid inputs for discrete bath object.");

                //load the continuous spectral density
                std::shared_ptr<continuous_bath<real_type>> cont = factory<continuous_bath<real_type>>::create(obj["continuous"]);

                //load the discretisation object.
                
                
                //now discretise the bath
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to load discrete bath from continuous bath object.");
            }
        }
        else if(IOWRAPPER::has_member(obj, "chain_mapped"))
        {
            try
            {
                chain_mapped_bath<real_type> chain(obj["chain_mapped"]);
                CALL_AND_HANDLE(chain.setup_orthopols(), "Failed to construct orthonormal polynomials.");
                m_g.resize(chain.poly().npoints());
                m_w.resize(m_g.size());
                for(size_t i = 0; i < m_g.size(); ++i)
                {
                    m_w[i] = chain.poly().node(i);
                    m_g[i] = std::sqrt(chain.poly().weight(i));
                }
                base_type::m_nonzero_temperature = false;
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to load discrete bath from chain mapped bath object.");
            }

            base_type::m_nterms = 1;
        }

        CALL_AND_HANDLE(IOWRAPPER::load_optional<real_type>(obj, "linewidth", m_jeps), "Failed to read in linewidth broadening.");
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to load discrete_bath bath from file.");
    } 
}
        

template class discrete_bath<double>;

}

#endif

