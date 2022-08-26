#ifndef HTUCKER_HAMILTONIAN_OPERATOR_HPP
#define HTUCKER_HAMILTONIAN_OPERATOR_HPP

#include "../operators/sum_of_product_operator.hpp"
#include "../ttn_nodes/operator_node.hpp"
#include "../tdvp_core/single_particle_operator.hpp"
#include "../tdvp_core/mean_field_operator.hpp"

namespace ttns
{

template <typename T, typename backend=linalg::blas_backend>
class hamiltonian
{
protected:
    using sop_type = sop_operator<T, backend>;

    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename backend::size_type;

    using spo_core = single_particle_operator_engine<T, backend>;
    using mfo_core = mean_field_operator_engine<T, backend>;

    using mat = linalg::matrix<T, backend>;
    using spfnode = operator_node_data<T, backend>;
    using index_array_type = typename spfnode::index_array_type;

protected:
    tree<mat> m_spo;
    tree<mat> m_mfo;
    tree<mat> m_sum;
    tree<mat> m_opA;
    tree<mat> m_temp;
    tree<bool> m_is_id;

public:
    hamiltonian() {}
    hamiltonian(const httensor<T, backend>& A) {CALL_AND_HANDLE(resize(A), "Failed to construct hamiltonian object.  Failed to allocate internal buffers.");}
    hamiltonian(const hamiltonian& o) = default;
    hamiltonian(hamiltonian&& o) = default;

    hamiltonian& operator=(const hamiltonian& o) = default;
    hamiltonian& operator=(hamiltonian&& o) = default;
    
    void resize(const httensor<T, backend>& A)
    {
        try
        {
            ASSERT(A.is_orthogonalised(), "The input hierarchical tucker tensor must have been orthogonalised.");
            CALL_AND_HANDLE(m_opA.resize(A), "Failed to resize the opa matrix tree.");
            CALL_AND_HANDLE(m_temp.resize(A), "Failed to resize the temporary matrix tree.");
            CALL_AND_HANDLE(m_spo.construct_topology(A), "Failed to construct the topology of the operator node tree.");
            CALL_AND_HANDLE(m_mfo.construct_topology(A), "Failed to construct the topology of the operator node tree.");
            CALL_AND_HANDLE(m_sum.construct_topology(A), "Failed to construct the topology of the operator node tree.");
            CALL_AND_HANDLE(m_is_id.construct_topology(A), "Failed to construct the topology of the bool node tree.");

            for(auto z : rzip(A, m_sum))
            {
                const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
                CALL_AND_HANDLE(spo_core::resize(a, hspf), "Failed to resize the matrix element buffers.");
            }

            for(auto z : rzip(A, m_spo))
            {
                const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
                CALL_AND_HANDLE(spo_core::resize(a, hspf), "Failed to resize the matrix element buffers.");
            }

            for(auto z : zip(A, m_mfo))
            {
                const auto& a = std::get<0>(z); auto& hmf = std::get<1>(z); 
                CALL_AND_HANDLE(mfo_core::resize(a, hmf), "Failed to resize the matrix element buffers.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize internal buffers of the hamiltonian object.");
        }
    }

    void operator()(const sop_type& h, httensor<T, backend>& psi)
    {
        try
        {
            ASSERT(psi.is_orthogonalised(), "The input hierarchical tucker tensor must have been orthogonalised.");
            ASSERT(has_same_structure(psi, m_spo), "The input hiearchical tucker tensor does not have the same topology as the hamiltonian object.");

            for(auto& s : m_sum)
            {
                s().fill_zeros();
            }

            for(size_type r=0; r<h.nterms(); ++r)
            {
                for(auto z : rzip(psi, m_spo, m_is_id, m_opA, m_temp))
                {
                    const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z); auto& is_id = std::get<2>(z); auto& ha = std::get<3>(z); auto& temp = std::get<4>(z);
                    CALL_AND_HANDLE(spo_core::evaluate(h, a, r, ha, temp, hspf, is_id), "Failed to evaluate single particle operator");
                }

                for(auto z : zip(psi, m_spo, m_is_id, m_mfo, m_opA, m_temp, m_sum))
                {
                    const auto& a = std::get<0>(z); const auto& hspf = std::get<1>(z); auto& is_id = std::get<2>(z); 
                    auto& hmf = std::get<3>(z); auto& ha = std::get<4>(z); auto& temp = std::get<5>(z); auto& s = std::get<6>(z);
                    if(!a.is_root())
                    {
                        CALL_AND_HANDLE(mfo_core::evaluate(hspf.parent(), is_id.parent(), a.parent(), ha.parent(), temp.parent(), hmf), "Failed to evaluate the mean field operator.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(mfo_core::set_root(hmf), "Failed to evaluate the root node of the mean field operator tree.");
                    }

                    if(is_id())
                    {
                        s() += h.coeff(r)*(hmf());
                    }
                    else
                    {
                        s() += h.coeff(r)*(hspf()*(hmf()));
                    }
                }
            }

            for(auto& s : m_sum)
            {
                std::cerr << s.id() << trace(s()) << std::endl;
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing inner product of hierarchical tucker tensor with itself.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of hierarchical tucker tensor with itself.");
        }
    }
};

}   //namespace ttns

#endif  //HTUCKER_HAMILTONIAN_OPERATOR_HPP//

