#ifndef HTUCKER_SINGLE_PARTICLE_OPERATOR_HPP
#define HTUCKER_SINGLE_PARTICLE_OPERATOR_HPP

#include "../operators/sum_of_product_operator.hpp"
#include "../ttn_nodes/operator_node.hpp"
#include "../tdvp_core/single_particle_operator.hpp"

namespace ttns
{

template <typename T, typename backend=linalg::blas_backend>
class single_particle_operator
{
protected:
    using sop_type = sop_operator<T, backend>;

    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename backend::size_type;

    using spf_core = single_particle_operator_engine<T, backend>;
    using spfnode = operator_node_data<T, backend>;
    using index_array_type = typename spfnode::index_array_type;

protected:
    tree<spfnode> m_hspf;
    tree<linalg::matrix<T, backend> > m_opA;
    tree<linalg::matrix<T, backend> > m_temp;

public:
    single_particle_operator() {}
    single_particle_operator(const sop_type& h, const httensor<T, backend>& A) {CALL_AND_HANDLE(initialise(h, A), "Failed to construct single_particle_operator object.  Failed to allocate internal buffers.");}
    single_particle_operator(const single_particle_operator& o) = default;
    single_particle_operator(single_particle_operator&& o) = default;

    single_particle_operator& operator=(const single_particle_operator& o) = default;
    single_particle_operator& operator=(single_particle_operator&& o) = default;
    
    void initialise(const sop_type& h, const httensor<T, backend>& A)
    {
        try
        {
            using utils::zip;   using utils::rzip;
            CALL_AND_HANDLE(m_opA.resize(A), "Failed to resize the opa matrix tree.");
            CALL_AND_HANDLE(m_temp.resize(A), "Failed to resize the temporary matrix tree.");
            CALL_AND_HANDLE(m_hspf.construct_topology(A), "Failed to construct the topology of the operator node tree.");
            index_array_type inds(h.nterms()); 
            for(auto z : rzip(A, m_hspf))
            {
                const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
                CALL_AND_HANDLE(spf_core::resize(h, a, hspf, inds), "Failed to resize the matrix element buffers.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize internal buffers of the single_particle_operator object.");
        }
    }

    T operator()(const sop_type& h, const httensor<T, backend>& psi)
    {
        try
        {
            using utils::zip;   using utils::rzip;
            ASSERT(has_same_structure(psi, m_hspf), "The input hiearchical tucker tensor does not have the same topology as the single_particle_operator object.");
            for(auto z : rzip(psi, m_hspf, m_opA, m_temp))
            {
                const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z); auto& ha = std::get<2>(z); auto& temp = std::get<3>(z);
                CALL_AND_HANDLE(spf_core::evaluate(h, a, ha, temp, hspf), "Failed to evaluate single particle opeartor");
            }
            T ret(0);
            for(size_type r=0; r<h.nterms(); ++r)
            {
                ret += h.coeff(r)*m_hspf[0]()[r](0, 0);
            }
            return ret;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute inner product of hierarchical tucker tensor with itself.");
        }
    }

};

}   //namespace ttns

#endif  //HTUCKER_MATRIX_ELEMENT_HPP//

