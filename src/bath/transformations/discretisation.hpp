#ifndef EOS_ABSTRACT_DISCRETISATION_HPP
#define EOS_ABSTRACT_DISCRETISATION_HPP

#include <array>

#include "../bath_types/bath.hpp"
#include "dscretisation_utilties.hpp"

namespace eos
{

template <typename T>
class abstract_discretisation
{
public:
    using real_type = T;
    using complex_type = linalg::complex<real_type>;
public:
    abstract_discretisation() : m_has_frequency_bounds(false), m_has_itol(false) {}
    abstract_discretisation(size_t nterms) : m_nterms(nterms){}
    abstract_discretisation(const abstract_discretisation& o) = default;
    abstract_discretisation(abstract_discretisation&& o) = default;
    virtual ~abstract_discretisation(){};

    virtual std::shared_ptr<abstract_discretisation<value_type>> clone() const = 0;
    
    virtual void print() = 0;
    virtual void load(const rapidjson::Value& obj) = 0;

    virtual void discretise(std::shared_ptr<bath<T>> bath, size_t N, std::shared_ptr<discrete_star<T>> disc) = 0;

    void set_itol(real_type itol) {m_has_itol = true;   m_itol = itol;}
    void set_frequency_bounds(const std::array<real_type, 2>& fb) {m_has_frequency_bounds = true;   m_itol = itol;}
protected:
    bool m_has_frequency_bounds;
    bool m_has_itol;

    T m_itol;
    T m_frequency_bounds;

    size_t m_integ_iterations;

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
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to load abstract_bath object from file.");
        }
    }

    abstract_discretisation(const rapidjson::Value& obj, std::string bname){CALL_AND_RETHROW(load(obj, bname));}
};

}

#endif

