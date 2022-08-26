#ifndef HTUCKER_SINGLE_PARTICLE_OPERATOR_CORE_HPP
#define HTUCKER_SINGLE_PARTICLE_OPERATOR_CORE_HPP

#include "kronecker_product_operator_helper.hpp"

namespace ttns
{

template <typename T, typename backend>
class single_particle_operator_engine
{
    using hnode = httensor_node<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using triad = std::vector<mat>;
    using matnode = typename tree<mat>::node_type;

    using spftype = operator_node_data<T, backend>;
    using spfnode = typename tree<spftype>::node_type;

    using boolnode = typename tree<bool>::node_type;

    using optype = sop_operator<T, backend>;
    using size_type = typename backend::size_type;

public:
    static inline void evaluate(const optype& h, const hnode& A, triad& HA, triad& temp, spfnode& hspf)
    {
        try
        {
            const auto& a = A().as_matrix();  

            //resize the matrices in the event that the tensor objects have changed size
            //CALL_AND_HANDLE(hspf().resize_matrices(A().size(1), A().size(1)), "Failed to resize the single-particle Hamiltonian operator matrices.");

            if(A.is_leaf())
            {
                //#pragma omp parallel for default(shared) schedule(dynamic, 1)
                for(size_type ind = 0; ind < hspf().nterms(); ++ind)
                {
                    if(!hspf()[ind].is_identity_spf())
                    {
                        //if there is only one spf term, then are handling either a common spf term or a 
                        //standard term and we just go ahead and evaluate the spf term
                        if(hspf()[ind].nspf_terms() == 1)
                        {
                            size_type ti = omp_get_thread_num();
                            auto& indices = hspf()[ind].spf_indexing()[0][0];
                            CALL_AND_HANDLE(h(indices[0], indices[1]).apply(a, HA[ti]), "Failed to apply leaf operator.");
                            CALL_AND_HANDLE(hspf()[ind].spf() = adjoint(a)*HA[ti], "Failed to apply matrix product to obtain result.");
                        }
                        else
                        {
                            hspf()[ind].spf().fill_zeros();
                            size_type ti = omp_get_thread_num();
                            for(size_type i = 0; i < hspf()[ind].nspf_terms(); ++i)
                            {
                                auto& indices = hspf()[ind].spf_indexing()[i][0];
                                CALL_AND_HANDLE(h(indices[0], indices[1]).apply(a, HA[ti]), "Failed to apply leaf operator.");
                                CALL_AND_HANDLE(hspf()[ind].spf() += hspf()[ind].accum_coeff(i)*adjoint(a)*HA[ti], "Failed to apply matrix product to obtain result.");
                            }
                        }
                    }
                }
            }
            else
            {
                using kpo = kronecker_product_operator<T, backend>;
                //#pragma omp parallel for default(shared) schedule(dynamic, 1)
                for(size_type ind = 0; ind < hspf().nterms(); ++ind)
                {
                    if(!hspf()[ind].is_identity_spf())
                    {
                        if(hspf()[ind].nspf_terms() == 1)
                        {
                            size_type ti = omp_get_thread_num();
                            CALL_AND_HANDLE(kpo::apply(hspf, ind, 0, A(), temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
                            CALL_AND_HANDLE(hspf()[ind].spf() = adjoint(a)*HA[ti], "Failed to apply matrix product to obtain result.");
                        }
                        else
                        {
                            hspf()[ind].spf().fill_zeros();
                            size_type ti = omp_get_thread_num();
                            for(size_type i = 0; i < hspf()[ind].nspf_terms(); ++i)
                            {
                                CALL_AND_HANDLE(kpo::apply(hspf, ind, i, A(), temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
                                CALL_AND_HANDLE(hspf()[ind].spf() += hspf()[ind].accum_coeff(i)*adjoint(a)*HA[ti], "Failed to apply matrix product to obtain result.");
                            }
                        }
                    }
                }
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating single particle operator at a node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate single particle operator at a node.");
        }
    }

};  //class single_particle_operator engine

}   //namespace ttns

#endif  //HTUCKER_SINGLE_PARTICLE_OPERATOR_CORE_HPP//
