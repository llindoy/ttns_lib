#ifndef EOS_BATH_HPP
#define EOS_BATH_HPP

#include <limits>
#include <memory>
#include <linalg/linalg.hpp>
#include "../utils/io.hpp"
#include "../utils/common.hpp"
#include "../utils/factory.hpp"
#include "../utils/quadrature/adaptive_integrate.hpp"

#include "../transformations/discretisation_utilities.hpp"

namespace eos
{
namespace bath
{

template <typename value_type> class continuous_bath;
template <typename value_type> class discrete_bath;
template <typename value_type> class chain_mapped_bath;

template <typename value_type> 
class abstract_bath
{
public:
    using real_type = value_type;
    using complex_type = linalg::complex<value_type>;

    using gauss_integ_type = quad::gauss::legendre<real_type>;
    using fourier_integ_type = quad::adaptive_fourier_integrals<real_type>;
public:
    abstract_bath() : m_nterms(1) {}
    abstract_bath(size_t nterms) : m_nterms(nterms){}
    abstract_bath(const abstract_bath& o) = default;
    abstract_bath(abstract_bath&& o) = default;
    virtual ~abstract_bath(){};

    virtual std::shared_ptr<abstract_bath<value_type>> clone() const = 0;
    
    virtual void print() = 0;
    virtual void load(const rapidjson::Value& obj) = 0;
    
    /*
     *  Functions for evaluating the spectral density
     */
    virtual real_type J(real_type w) const
    {
        ASSERT(m_nterms == 1, "Need to specify index when working with correlated bath.");
        CALL_AND_RETHROW(return this->J(w, 0, 0));
    }
    virtual real_type J(real_type w, size_t i, size_t j) const = 0;                                         //J(w
    virtual void J(real_type w, linalg::matrix<real_type>& Jv) const
    {
        ASSERT(Jv.shape(0) == m_nterms && Jv.shape(1) == m_nterms, "Failed to initial spectral density matrix.  Invalid size.");
        for(size_t i = 0; i < m_nterms; ++i)
        {
            for(size_t j = 0; j < m_nterms; ++j)
            {
                CALL_AND_RETHROW(Jv(i, j) = this->J(w, i, j));
            }
        }
    }                                   

    /*
     *  Functions for evaluating the noise power spectrum for a bosonic bath
     */
    virtual real_type S(real_type w) const
    {
        ASSERT(m_nterms == 1, "Need to specify index when working with correlated bath.");
        CALL_AND_RETHROW(return this->S(w, 0, 0));
    }
    virtual real_type S(real_type w, size_t i, size_t j) const = 0;

    /*
     *  Functions for evaluating the bath correlation function
     */
    //functions for when we have analytic bath correlation functions
    virtual complex_type C(real_type t) const
    {
        ASSERT(m_nterms == 1, "Need to specify index when working with correlated bath.");
        CALL_AND_RETHROW(return this->C(t, 0, 0));
    } 
    virtual complex_type C(real_type, size_t, size_t) const{RAISE_EXCEPTION("Analytic bath correlation functions have not been implemented for this system.");}
    virtual void C(real_type t, linalg::matrix<complex_type>& Cv) const
    {
        ASSERT(Cv.shape(0) == m_nterms && Cv.shape(1) == m_nterms, "Failed to initial spectral density matrix.  Invalid size.");
        for(size_t i = 0; i < m_nterms; ++i)
        {
            for(size_t j = 0; j < m_nterms; ++j)
            {
                CALL_AND_RETHROW(Cv(i, j) = this->C(t, i, j));
            }
        }
    }

    //function for numerical evaluation of the bath correlation function
    virtual complex_type C(real_type t, fourier_integ_type& integ) const
    {
        ASSERT(m_nterms == 1, "Need to specify index when working with correlated bath.");
        CALL_AND_RETHROW(return this->C(t, 0, 0, integ));
    }
    virtual complex_type C(real_type t, size_t i, size_t j, fourier_integ_type& integ) const 
    {
        try
        {
            if(this->nonzero_temperature())
            {
                real_type Cr = integ.cosine([&](real_type w){return this->S(w, i, j);}, t, 0, this->integral_upper_bound(), this->trial_upper_bound());
                real_type Ci = integ.sine([&](real_type w){return this->J(w, i, j);}, t, 0, this->integral_upper_bound(), this->trial_upper_bound());
                return complex_type(Cr, -Ci);
            }
            else
            {
                return integ.fourier([&](real_type w){return this->J(w, i, j);}, t, 0, this->integral_upper_bound(), this->trial_upper_bound());
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute bath correlation function.");
        }
    }     
    virtual void C(real_type t, linalg::matrix<complex_type>& Cv, fourier_integ_type& integ) const 
    {
        ASSERT(Cv.shape(0) == m_nterms && Cv.shape(1) == m_nterms, "Failed to initial spectral density matrix.  Invalid size.");
        for(size_t i = 0; i < m_nterms; ++i)
        {
            for(size_t j = 0; j < m_nterms; ++j)
            {
                CALL_AND_RETHROW(Cv(i, j) = this->C(t, i, j, integ));
            }
        }
    }


    /*
     * Functions for evaluating the etas that appear in the numerical 
     */
    virtual complex_type eta(real_type t, fourier_integ_type& integ) const
    {
        ASSERT(m_nterms == 1, "Need to specify index when working with correlated bath.");
        CALL_AND_RETHROW(return this->eta(t, 0, 0, integ));
    }
    virtual complex_type eta(real_type t, size_t i, size_t j, fourier_integ_type& integ) const
    {
        try
        {
            real_type etai = integ.integrate(
                [&](real_type w)
                {   
                    return this->J(w, i, j)*eta_imag_factor(w, t);
                }, 
                0, this->integral_upper_bound(), this->trial_upper_bound());

            real_type etar;
            if(this->nonzero_temperature())
            {
                etar = integ.integrate(
                    [&](real_type w)
                    {
                        return this->S(w, i, j)*eta_real_factor(w, t);
                    }, 
                    0, this->integral_upper_bound(), this->trial_upper_bound());
            }
            else
            {            
                etar = integ.integrate(
                    [&](real_type w)
                    {
                        return this->J(w, i, j)*eta_real_factor(w, t);
                    }, 
                    0, this->integral_upper_bound(), this->trial_upper_bound());
            }
            return complex_type(etar, etai)/M_PI;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute eta function.");
        }
    }
    virtual void eta(real_type t, linalg::matrix<complex_type>& etav, fourier_integ_type& integ) const
    {
        ASSERT(etav.shape(0) == m_nterms && etav.shape(1) == m_nterms, "Failed to initial spectral density matrix.  Invalid size.");
        for(size_t i = 0; i < m_nterms; ++i)
        {
            for(size_t j = 0; j < m_nterms; ++j)
            {
                CALL_AND_RETHROW(etav(i, j) = this->eta(t, i, j, integ));
            }
        }
    }

    virtual complex_type eta(size_t k, size_t kp, size_t N, real_type dt, fourier_integ_type& integ) const
    {
        ASSERT(m_nterms == 1, "Need to specify index when working with correlated bath.");
        CALL_AND_RETHROW(return this->eta(k, kp, N, dt, 0, 0, integ));
    }
    virtual complex_type eta(size_t k, size_t kp, size_t N, real_type dt, size_t i, size_t j, fourier_integ_type& integ) const
    {   
        try
        {
            ASSERT(k >= kp, "Invalid ordering for eta.");
            if(k == kp)
            {
                if(k == 0 || k == N)
                {
                    return eta(dt/2.0, i, j, integ);
                }
                else
                {
                    return eta(dt, i, j, integ);
                }
            }
            else
            {
                return (this->eta((k-kp+1)*dt, i, j, integ) - 2.0*this->eta((k-kp)*dt, i, j, integ) + this->eta((k-kp-1)*dt, i, j, integ));
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute discretised eta.");
        }
    }
    virtual void eta(size_t k, size_t kp, size_t N, real_type dt, linalg::matrix<complex_type>& etav, fourier_integ_type& integ) const
    {
        ASSERT(etav.shape(0) == m_nterms && etav.shape(1) == m_nterms, "Failed to initial spectral density matrix.  Invalid size.");
        for(size_t i = 0; i < m_nterms; ++i)
        {
            for(size_t j = 0; j < m_nterms; ++j)
            {
                CALL_AND_RETHROW(etav(i, j) = this->eta(k, kp, N, dt, i, j, integ));
            }
        }
    }

    virtual bool is_continuous() const{return false;}
    virtual bool is_discrete() const{return false;}
    virtual bool is_chain_mapped() const{return false;}

    virtual std::shared_ptr<continuous_bath<value_type>> as_continuous() const{RAISE_EXCEPTION("Cannot convert to continuous bath.");}
    virtual std::shared_ptr<discrete_bath<value_type>> as_discrete() const{RAISE_EXCEPTION("Cannot convert to discrete bath.");}
    virtual std::shared_ptr<chain_mapped_bath<value_type>> as_chain_mapped() const{RAISE_EXCEPTION("Cannot convert to chain mapped bath.");}

    void set_temperature(real_type T)
    {
        if(std::abs(T) < 1e-14){m_nonzero_temperature = false;}
        else{m_beta = 1.0/T; m_nonzero_temperature = true;}
    }
    void set_beta(real_type beta)
    {
        m_beta = beta; m_nonzero_temperature = true;
    }

    bool nonzero_temperature() const{return m_nonzero_temperature;}
    real_type T() const
    {
        if(!m_nonzero_temperature){return 1.0/m_beta;}
        else{return 0.0;}
    }

    real_type beta() const
    {
        if(!m_nonzero_temperature){return std::numeric_limits<real_type>::infinity();}
        return m_beta;
    }
    
protected:
    void load(const rapidjson::Value& obj, std::string bname)
    {
        try
        {
            ASSERT(obj.IsObject(), "Invalid rapidjson object.");
            ASSERT(obj.HasMember("type"), "Object does not specify a spectral_density type.");
            ASSERT(obj["type"].IsString(), "The spectral density type is not a string");
            std::string s(obj["type"].GetString());
            remove_whitespace_and_to_lower(s);
            remove_whitespace_and_to_lower(bname);
            ASSERT(s == bname, "The input spectral density type differs from the type that is being created.");
            if(obj.HasMember("beta"))
            {
                ASSERT(!obj.HasMember("t"), "Cannot specify both a temperature and inverse temperature.");
                ASSERT(obj["beta"].IsNumber(), "Bath Temperature Info Found but is invalid.");
                m_beta = obj["beta"].GetDouble();
                ASSERT(m_beta >= 0, "Invalid inverse temperature.");
                m_nonzero_temperature = true;
            }
            else if(obj.HasMember("t"))
            {
                ASSERT(obj["t"].IsNumber(), "Bath Temperature Info Found but is invalid.");
                real_type T = obj["t"].GetDouble();
                ASSERT(T >= 0, "Invalid temperature.");
                if(std::abs(T) > 1e-14)
                {
                    m_nonzero_temperature = true;
                    m_beta = 1.0/T;
                }
                else
                {
                    m_nonzero_temperature = false;
                }
            }
            else
            {
                m_nonzero_temperature = false;
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to load abstract_bath object from file.");
        }
    }
    abstract_bath(const rapidjson::Value& obj, std::string bname){CALL_AND_RETHROW(load(obj, bname));}

    size_t nterms() const{return m_nterms;}
protected:
    size_t m_nterms;
    bool m_nonzero_temperature;
    bool m_use_thermofield;
    real_type m_beta;

protected:
    virtual real_type integral_upper_bound() const{return std::numeric_limits<real_type>::infinity();};
    virtual real_type trial_upper_bound() const = 0;

    // (cos(wt)-1)/w^2
    static real_type eta_real_factor(real_type w, real_type t)
    {
        if(std::abs(w) > 1e-6){return (1.0 - std::cos(w*t))/(w*w);} 
        else
        {
            real_type factor = t*t/2.0;
            real_type res = 0.0;
            for(size_t k=0; k < 40; ++k)
            {
                res += factor;
                factor *= -1.0*(w*w*t*t)/(2*k+3)*(2*k+4);
            }
            res += factor;
            return res;
        }
    }

    static real_type eta_imag_factor(real_type w, real_type t)
    {
        if(std::abs(w) > 1e-6){return (std::sin(w*t) - w*t)/(w*w);} 
        else
        {
            real_type factor = -w*t*t*t/6.0;
            real_type res = 0.0;
            for(size_t k=0; k < 40; ++k)
            {
                res += factor;
                factor *= -1.0*(w*w*t*t)/(2*k+4)*(2*k+5);
            }
            res += factor;
            return res;
        }
    }
};

template <typename value_type> 
class continuous_bath : public abstract_bath<value_type>
{
public:
    using base_type = abstract_bath<value_type>;
    using real_type = typename base_type::real_type;
    using complex_type = typename base_type::complex_type;
    using fourier_integ_type = typename base_type::fourier_integ_type;
    using gauss_integ_type = typename base_type::gauss_integ_type;
public:
    continuous_bath() : base_type() {}
    continuous_bath(size_t nterms) : base_type(nterms){}
    continuous_bath(const continuous_bath& o) = default;
    continuous_bath(continuous_bath&& o) = default;
    virtual ~continuous_bath(){}

    /*
     *  Functions used for the density based algorithm for discretising the spectral density
     */
    virtual real_type density_of_frequencies(real_type) const{RAISE_EXCEPTION("Density based discretisation is not supported for this bath type.");}
    virtual real_type density_of_frequencies(real_type, real_type) const{RAISE_EXCEPTION("Density based discretisation is not supported for this bath type.");}

    bool is_continuous() const final{return true;}

    virtual std::array<real_type, 2> frequency_bounds(real_type, const gauss_integ_type&, bool /*use_thermofield  */ = false) const{RAISE_EXCEPTION("Density based discretisation is not supported for this bath type.");}
    /*
     *  Functions for constructing polynomials that are orthonormal with respect to this spectral density.  
     */
    //virtual void classical_orthonormal_polynomials(orthopol& poly) const{RAISE_EXCEPTION("Classical Orthonormal Polynomial Quadrature is not supported for this bath type.");}
    //virtual void nonclassical_orthonormal_polynomials(orthopol& poly, const real_type& wbound) const{RAISE_EXCEPTION("Nonclassical Orthonormal Polynomial Quadrature is not supported for this bath type.");}
};

}
}   //namespace eos

#endif

