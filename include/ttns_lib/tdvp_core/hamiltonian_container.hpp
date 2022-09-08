#ifndef TTNS_HAMILTONIAN_CONTAINER_HPP
#define TTNS_HAMILTONIAN_CONTAINER_HPP

#include "../ttn_nodes/operator_node.hpp"

#include <vector>
#include <list>
#include <algorithm>

namespace ttns
{
template <typename T, typename backend = linalg::blas_backend>
class hamiltonian_container
{
public:
    using opdata_type = operator_node_data<T, backend>;
    using operator_type = sop_operator<T, backend>;
    using index_array_type = typename opdata_type::index_array_type;
    using spo_core = single_particle_operator_engine<T, backend>;
    using mfo_core = mean_field_operator_engine<T, backend>;
    using mat_type = linalg::matrix<T, backend>;

protected:
    tree<opdata_type> m_spo;
    tree<opdata_type> m_mfo;

public:
    hamiltonian_container() {}
    hamiltonian_container(const httensor<T, backend>& A, const operator_type& op, bool use_sum = false) 
    {
        try
        {
            CALL_AND_HANDLE(resize(A, op, use_sum), "Call to initialisation failed.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct hamiltonian container object.");
        }
    }   
    hamiltonian_container(const hamiltonian_container& o)
    {
        try
        {
            CALL_AND_HANDLE(m_spo = o.m_spo, "Failed to copy single particle operator tree.");
            CALL_AND_HANDLE(m_mfo = o.m_mfo, "Failed to copy mean field operator tree.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to copy construct hamiltonian container object.");
        }
    }
    template <typename be, typename = typename std::enable_if<not std::is_same<backend, be>::value, void>::type>
    hamiltonian_container(const hamiltonian_container<T, be>& o)
    {
        try
        {
            CALL_AND_HANDLE(m_spo = o.m_spo, "Failed to copy single particle operator tree.");
            CALL_AND_HANDLE(m_mfo = o.m_mfo, "Failed to copy mean field operator tree.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to copy construct hamiltonian container object.");
        }
    }
    hamiltonian_container(hamiltonian_container&& o) = default;


    //copy assignment operators
    hamiltonian_container& operator=(const hamiltonian_container& o)
    {
        try
        {
            if(&o != this)
            {
                CALL_AND_HANDLE(m_spo = o.m_spo, "Failed to copy single particle operator tree.");
                CALL_AND_HANDLE(m_mfo = o.m_mfo, "Failed to copy mean field operator tree.");
            }
            return *this;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to copy assign hamiltonian container object.");
        }
    }
    template <typename be, typename = typename std::enable_if<not std::is_same<backend, be>::value, void>::type>
    hamiltonian_container& operator=(const hamiltonian_container<T, be>& o)
    {
        try
        {
            CALL_AND_HANDLE(m_spo = o.m_spo, "Failed to copy single particle operator tree.");
            CALL_AND_HANDLE(m_mfo = o.m_mfo, "Failed to copy mean field operator tree.");

            return *this;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to copy assign hamiltonian container object.");
        }
    }

    hamiltonian_container& operator=(hamiltonian_container&& o) = default;

    void resize(const httensor<T, backend>& A, const operator_type& op, bool use_sum = false)
    {
        try
        {
            CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");

            //now we resize all of the objects necessary to compute the Hamiltonian operators at each node
            CALL_AND_HANDLE(m_spo.construct_topology(A), "Failed to construct the topology of the operator node tree.");
            CALL_AND_HANDLE(m_mfo.construct_topology(A), "Failed to construct the topology of the operator node tree.");
            using utils::zip;   using utils::rzip;

            index_array_type inds;  CALL_AND_HANDLE(inds.resize(op.nterms()), "Failed to resize the temporary inds array.");
            for(auto z : rzip(A, m_spo))
            {
                const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
                CALL_AND_HANDLE(spo_core::resize(op, a, hspf, inds, use_sum), "Failed to resize the single particle operator tree nodes.");
            }

            for(auto z : zip(A, m_spo, m_mfo))
            {
                const auto& a = std::get<0>(z); const auto& hspf = std::get<1>(z); auto& hmf = std::get<2>(z);
                CALL_AND_HANDLE(mfo_core::resize(hspf, a, hmf, inds, use_sum), "Failed to resize the mean field operator tree nodes.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize the hamiltonian_container object.");
        }
    }
    template <typename U, typename be>
    void resize(const hamiltonian_container<U, be>& o)
    {
        try
        {
            CALL_AND_HANDLE(m_spo.resize(o.m_spo), "Failed to resize single particle operator tree.");
            CALL_AND_HANDLE(m_mfo.resize(o.m_mfo), "Failed to copy mean field operator tree.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize the hamiltonian container object.");
        }
    }

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_spo.clear(), "Failed to clear the single particle operator tree.");
            CALL_AND_HANDLE(m_mfo.clear(), "Failed to clear the mean field operator tree.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the hamiltonian_container object.");
        }
    }   
    
    tree<opdata_type>& single_particle_operator(){return m_spo;}
    const tree<opdata_type>& single_particle_operator() const {return m_spo;}

    tree<opdata_type>& mean_field_operator(){return m_mfo;}
    const tree<opdata_type>& mean_field_operator() const {return m_mfo;}

    bool validate_size(const httensor<T, backend>& A, const operator_type& op)
    {
        if(!has_same_structure(A, m_spo)){return false;}
        else
        {
            using utils::zip;   using utils::rzip;
            index_array_type inds;  CALL_AND_HANDLE(inds.resize(op.nterms()), "Failed to resize the temporary inds array.");
            for(auto z : rzip(A, m_spo))
            {
                bool has_same_size = true;
                const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
                CALL_AND_HANDLE(has_same_size = spo_core::check_size(op, a, hspf, inds), "Failed to resize the single particle operator tree nodes.");
                if(!has_same_size){return false;}
            }

            for(auto z : zip(A, m_spo, m_mfo))
            {
                bool has_same_size = true;
                const auto& a = std::get<0>(z); const auto& hspf = std::get<1>(z); auto& hmf = std::get<2>(z);
                CALL_AND_HANDLE(has_same_size = mfo_core::check_size(hspf, a, hmf, inds), "Failed to resize the mean field operator tree nodes.");
                if(!has_same_size){return false;}
            }
            return true;
        }
    }


#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void serialize(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("single_particle_operator", m_spo)), "Failed to serialise hamiltonian container.  Failed to serialise the single particle operator tree.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("mean_field_operator", m_mfo)), "Failed to serialise hamiltonian container.  Failed to serialise the mean field operator tree.");
    }
#endif
};  //hamiltonian_container


}   //namespace ttns

#endif  //TTNS_HAMILTONIAN_CONTAINER_HPP

