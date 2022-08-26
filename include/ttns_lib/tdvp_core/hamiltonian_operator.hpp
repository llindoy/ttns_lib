#ifndef HTUCKER_MEAN_FIELD_OPERATOR_CORE_HPP
#define HTUCKER_MEAN_FIELD_OPERATOR_CORE_HPP


namespace ttns
{

namespace hamiltonian_helper
{
template <typename T, typename backend>
class rank4_contraction;

template <typename  T>
class rank4_contraction<T, linalg::blas_backend>
{
    using backend = linalg::blas_backend;
    using hnode = httensor_node<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using size_type = typename backend::size_type;
public:
    static void apply(const hnode& A, const hnode& B, mat& C, size_type mode)
    {
        try
        {
            size_type m = A().dim(mode);
            size_type n = A().hrank();

            auto _A = A().as_rank_4(mode);
            auto _B = B().as_rank_4(mode);
            auto _C = C.reinterpret_shape(m, m, n, n);

            ASSERT(_C.shape(0) == _A.shape(1) && _C.shape(2) == _A.shape(3), "Invalid array shapes.");

            for(size_t i = 0; i < m; ++i)
            {
                for(size_t j = 0; j < m; ++j)
                {
                    for(size_t k = 0; k < n; ++k)
                    {
                        for(size_t l = 0; l < n; ++l)
                        {
                            T temp(0);
                            for(size_t I1 = 0; I1 < _A.shape(0); ++I1)
                            {
                                for(size_t I2 = 0; I2 < _A.shape(2); ++I2)
                                {
                                    temp += conj(_A(I1, i, I2, k))*_B(I1, j, I2, l);
                                }
                            }
                            _C(i,j,k,l) = temp;
                        }
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate rank4 contraction useful in the evaluatino of the mean field operators.");
        }
    }
};
}

template <typename T, typename backend>
class hamiltonian_operator_engine
{
    using hnode = httensor_node<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using triad = std::vector<mat>;
    using matnode = typename tree<mat>::node_type;

    using hamtype = hamiltonian_node_data<T, backend>;
    using hamnode = typename tree<hamtype>::node_type;

    using boolnode = typename tree<bool>::node_type;

    using size_type = typename backend::size_type;
    using optype = sop_operator<T, backend>;

public:

    static inline size_type contraction_buffer_size(const hnode& A)
    {
        size_type maxdim = 0;
        for(size_type mode = 0; mode < A().nmodes(); ++mode)
        {
            auto _A = A().as_rank_3(mode);
            size_type dim = _A.shape(0)*_A.shape(1)*_A.shape(1);
            if(dim > maxdim){maxdim = dim;}
        }
        return maxdim;
    }

    template <typename vec, typename v2>
    static inline void evaluate_mfo(const hnode& A, triad& HA, triad& temp, vec& temp2, mat& Cijkl, const v2& coeff, hamnode& hmf)
    {
        try
        {
            ASSERT(hmf().accumulate_terms(), "Invalid hmf setup for tdvp integrator.");
            //we only need to update the mean field Hamiltonian if it isn't the root node.
            if(!hmf.is_root())
            {
                INIT_TIMER;
                INIT_TIMER_T;
                START_TIMER_T;
                const auto& hmf_p = hmf.parent();

                ASSERT(hmf().nterms() == hspf().nterms(), "The number of terms in the single particle and mean field operators are not the same.");
                ASSERT(A().hrank() == hmf_p().dimen(), "The parent node of the mean field operator and the input hierarchical tucker tensor are not the same size.");

                size_type mode = hmf.child_id();


                size_type n_identity_spf = 0;
                size_type n_contracted = 0;
                START_TIMER;
                bool parent_has_hmf_contract = false;
                for(size_type r=0; r<hmf().nterms(); ++r)
                {       
                    if(!hmf().is_identity(r))
                    {
                        if(all_spf_ops_identity(hspf, mode, r)){++n_identity_spf;}
                    }
                    if(!hmf().is_present(r)){++n_contracted;}
                    if(!hmf_p().is_present(r)){parent_has_hmf_contract = true;}
                }
                bool use_rank_4 = parent_has_hmf_contract || n_identity_spf > 0;
                STOP_TIMER("docalc");
                STOP_TIMER_T("docalc_T");

                START_TIMER;
                if(use_rank_4)
                {
                    //first we resize and evaluate the tensor with elements Cijkl = \sum_{I_1, I_2} A^*_{I_1, i, I_2, k} A_{I_1, j, I_2, l} 
                    Cijkl.resize(A().dim(mode)*A().dim(mode), A().hrank()*A().hrank());
                    using r4_contract = mfo_helper::rank4_contraction<T, backend>;
                    CALL_AND_HANDLE(r4_contract::apply(A, A, Cijkl, mode), "Failed to evaluate rank4 contraction of A tensor.");
                }
                STOP_TIMER("Evalrank4");
                STOP_TIMER_T("Evalrank4_T");

                //check whether the parent node has any hmf_contract terms.  If it does we evaluate their contribution to the current nodes
                //hmf_contract term.  As the parent nodes have an identity operator, it means that all of the identity operators that need
                //to be included at this level are also identity operators and so we only need to use the result of the rank 4 contraction.
                START_TIMER;
                if(parent_has_hmf_contract)
                {
                    try
                    {
                        auto _hmfv = hmf().accumulated().reinterpret_shape(hmf().accumulated().size());
                        auto _hmfpv = hmf_p().accumulated().reinterpret_shape(hmf_p().accumulated().size());
                        _hmfv = Cijkl*_hmfpv;
                    }
                    catch(const std::exception& ex)
                    {
                        std::cerr << ex.what() << std::endl;
                        RAISE_EXCEPTION("Failed to evaluate mfo contraction when the single particle operator si the identity.");
                    }
                }
                else if(n_contracted > 0){hmf().accumulated().fill_zeros();}
                STOP_TIMER("Contractparent");
                STOP_TIMER_T("contractparent_T");
                //#pragma omp parallel for default(shared) schedule(dynamic, 1)
                for(size_type r=0; r<hmf().nterms(); ++r)
                {       
                    size_type ti = omp_get_thread_num();
                    //we only update the mean field operator if this term is not the identity operator
                    if(!hmf().is_identity(r))
                    {
                        START_TIMER;
                        if(hmf().is_present(r))
                        {
                            //now we determine whether all single particle hamiltonians are the identity operator.  In this case we 
                            //can use the rank4 contraction evaluated 
                            if(all_spf_ops_identity(hspf, mode, r))
                            {
                                try
                                {
                                    auto _hmfv = hmf()[r].reinterpret_shape(hmf()[r].size());
                                    auto _hmfpv = hmf_p()[r].reinterpret_shape(hmf_p()[r].size());
                                    _hmfv = Cijkl*_hmfpv;
                                }
                                catch(const std::exception& ex)
                                {
                                    std::cerr << ex.what() << std::endl;
                                    RAISE_EXCEPTION("Failed to evaluate mfo contraction when the single particle operator si the identity.");
                                }
                                STOP_TIMER("allid");
                                STOP_TIMER_T("allid_T");
                            }
                            else
                            {
                                //act the single particle operator on the node
                                using kpo = kronecker_product_operator<T, backend>;
                                CALL_AND_HANDLE(kpo::apply(hspf, r, mode, A(), temp[ti], HA[ti]), "Failed to evaluate action of kronecker product operator.");

                                //now if the parent node is not an identity operator we actually need to act it on the A tensor
                                if(!hmf_p().is_identity(r))
                                {
                                    CALL_AND_HANDLE(temp[ti] = HA[ti]*trans(hmf_p()[r]), "Failed to apply action of parent mean field operator.");
                                    CALL_AND_HANDLE(HA[ti] = conj(A().as_matrix()), "Failed to compute conjugate of the A matrix.");

                                    try
                                    {
                                        auto _A = A().as_rank_3(mode);
                                        auto _HA = HA[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                                        auto _temp = temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                                        CALL_AND_HANDLE(hmf()[r] = (contract(_HA, 0, 2, _temp, 0, 2).bind_workspace(temp2)), "Failed when evaluating the final contraction.");
                                    }                            
                                    catch(const linalg::invalid_value& ex)
                                    {
                                        std::cerr << ex.what() << std::endl;
                                        RAISE_NUMERIC("forming temporary reinterpreted and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                    }
                                    catch(const std::exception& ex)
                                    {
                                        std::cerr << ex.what() << std::endl;
                                        RAISE_EXCEPTION("Failed to form temporary reinterpreted and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                    }
                                    STOP_TIMER("evalspos");
                                    STOP_TIMER_T("evalspos_T");
                                }
                                else
                                {
                                    CALL_AND_HANDLE(temp[ti] = conj(A().as_matrix()), "Failed to compute conjugate of the A matrix.");
            
                                    try
                                    {
                                        auto _A = A().as_rank_3(mode);
                                        auto _HA = HA[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                                        auto _temp = temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                                        CALL_AND_HANDLE(hmf()[r] = (contract(_temp, 0, 2, _HA, 0, 2).bind_workspace(temp2)), "Failed when evaluating the final contraction.");
                                    }
                                    catch(const linalg::invalid_value& ex)
                                    {
                                        std::cerr << ex.what() << std::endl;
                                        RAISE_NUMERIC("forming temporary reinterpreted and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                    }
                                    catch(const std::exception& ex)
                                    {
                                        std::cerr << ex.what() << std::endl;
                                        RAISE_EXCEPTION("Failed to form temporary reinterpreted and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                    }
                                    STOP_TIMER("identityspos");
                                    STOP_TIMER_T("identityspos_T");
                                }
                            }
                        }
                        //if this node has an identity operator for its single particle function but its parent does not, then we compute the mean field operator and 
                        //add it to the hmf().accumulated() matri.  We have treated the other possible branch of this if statement already
                        //TODO: Improve the Kronecker we can accumulate the action of the Hamiltonian on the tensor before evaluating the final contraction.
                        //This will reduce the number of large contractions that need to be performed.
                        else if(!hmf().is_present(r) && hmf_p().is_present(r))
                        {
                            //now we determine whether all single particle hamiltonians are the identity operator.  In this case we 
                            //can use the rank4 contraction evaluated 
                            if(all_spf_ops_identity(hspf, mode, r))
                            {
                                try
                                {
                                    auto _hmfv = hmf().accumulated().reinterpret_shape(hmf().accumulated().size());
                                    auto _hmfpv = hmf_p()[r].reinterpret_shape(hmf_p()[r].size());
                                    _hmfv += coeff[r]*Cijkl*_hmfpv;
                                }
                                catch(const std::exception& ex)
                                {
                                    std::cerr << ex.what() << std::endl;
                                    RAISE_EXCEPTION("Failed to evaluate mfo contraction when the single particle operator si the identity.");
                                }
                                    STOP_TIMER("allid2");
                                    STOP_TIMER_T("allid2_T");
                            }
                            else
                            {   
                                //act the single particle operator on the node
                                using kpo = kronecker_product_operator<T, backend>;
                                CALL_AND_HANDLE(kpo::apply(hspf, r, mode, A(), temp[ti], HA[ti]), "Failed to evaluate action of kronecker product operator.");

                                //now if the parent node is not an identity operator we actually need to act it on the A tensor
                                if(!hmf_p().is_identity(r))
                                {
                                    CALL_AND_HANDLE(temp[ti] = HA[ti]*trans(hmf_p()[r]), "Failed to apply action of parent mean field operator.");
                                    CALL_AND_HANDLE(HA[ti] = conj(A().as_matrix()), "Failed to compute conjugate of the A matrix.");

                                    try
                                    {
                                        auto _A = A().as_rank_3(mode);
                                        auto _HA = HA[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                                        auto _temp = temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                                        CALL_AND_HANDLE(hmf().accumulated() += coeff[r]*(contract(_HA, 0, 2, _temp, 0, 2).bind_workspace(temp2)), "Failed when evaluating the final contraction.");
                                    }                            
                                    catch(const linalg::invalid_value& ex)
                                    {
                                        std::cerr << ex.what() << std::endl;
                                        RAISE_NUMERIC("forming temporary reinterpreted and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                    }
                                    catch(const std::exception& ex)
                                    {
                                        std::cerr << ex.what() << std::endl;
                                        RAISE_EXCEPTION("Failed to form temporary reinterpreted and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                    }
                                    STOP_TIMER("evalspos2");
                                    STOP_TIMER_T("evalspos2_T");
                                }
                                else
                                {
                                    //we can speed this one up by first accumulating all of the mean field operators 
                                    CALL_AND_HANDLE(temp[ti] = conj(A().as_matrix()), "Failed to compute conjugate of the A matrix.");
            
                                    try
                                    {
                                        auto _A = A().as_rank_3(mode);
                                        auto _HA = HA[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                                        auto _temp = temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                                        CALL_AND_HANDLE(hmf().accumulated() += (coeff[r]*contract(_temp, 0, 2, _HA, 0, 2).bind_workspace(temp2)), "Failed when evaluating the final contraction.");
                                    }
                                    catch(const linalg::invalid_value& ex)
                                    {
                                        std::cerr << ex.what() << std::endl;
                                        RAISE_NUMERIC("forming temporary reinterpreted and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                    }
                                    catch(const std::exception& ex)
                                    {
                                        std::cerr << ex.what() << std::endl;
                                        RAISE_EXCEPTION("Failed to form temporary reinterpreted and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                    }
                                    STOP_TIMER("idspos2");
                                    STOP_TIMER_T("idspos2_T");
                                }
                            }
                        }
                    }
                }
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating mean field operator at a node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate mean field operator at a node.");
        }
    }
    
    static inline void evaluate_spo(const optype& h, const hnode& A, triad& HA, triad& temp, hamnode& ham)
    {
        try
        {
            const auto& a = A().as_matrix();  
            if(A.is_leaf())
            {
                size_type lid = A.leaf_index();

                #pragma omp parallel for default(shared) schedule(dynamic, 1)
                for(size_type ind = 0; ind < ham().nham_first_accumulate(); ++ind)
                {
                    size_type ti = omp_get_thread_num();
                    size_type r = ham().ham_first_acccumulate_indices(ind);
                    CALL_AND_HANDLE(h(r, lid)->apply(a, HA[ti]), "Failed to apply leaf operator.");
                    CALL_AND_HANDLE(ham().hspf_accumulated() += h.coeff(r)*adjoint(a)*HA[ti], "Failed to apply matrix product to obtain result.");
                }

                #pragma omp parallel for default(shared) schedule(dynamic, 1)
                for(size_type ind = 0; ind < ham().nni(); ++ind)
                {
                    size_type ti = omp_get_thread_num();
                    size_type r = ham().rindex(ind);
                    CALL_AND_HANDLE(h(r, lid)->apply(a, HA[ti]), "Failed to apply leaf operator.");
                    CALL_AND_HANDLE(ham().hspf(r) = adjoint(a)*HA[ti], "Failed to apply matrix product to obtain result.");
                }
            }
            else
            {
                //if any of the child nodes have an accumulated term, then we need to act them up to here and accumulate them together
                #pragma omp parallel for default(shared) schedule(dynamic, 1)
                for(size_type i = 0; i < h.size(); ++i)
                {
                    if(
                }

                #pragma omp parallel for default(shared) schedule(dynamic, 1)
                for(size_type ind = 0; ind < ham().nni(); ++ind)
                {
                    size_type ti = omp_get_thread_num();
                    size_type r = ham().rindex(ind);
                    using kpo = kronecker_product_operator<T, backend>;
                    CALL_AND_HANDLE(kpo::apply(ham, r, A(), temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
                    CALL_AND_HANDLE(ham()[r] = adjoint(a)*HA[ti], "Failed to apply matrix product to obtain result.");
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

};  //class mean field operator engine

}   //namespace ttns

#endif  //HTUCKER_MEAN_FIELD_OPERATOR_CORE_HPP//

