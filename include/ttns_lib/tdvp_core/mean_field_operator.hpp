#ifndef HTUCKER_MEAN_FIELD_OPERATOR_CORE_HPP
#define HTUCKER_MEAN_FIELD_OPERATOR_CORE_HPP

namespace ttns
{

template <typename T, typename backend>
class mean_field_operator_engine
{
    using hnode = httensor_node<T, backend>;
    using hdata = httensor_node_data<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using triad = std::vector<mat>;
    using matnode = typename tree<mat>::node_type;

    using optype = operator_node_data<T, backend>;
    using opnode = typename tree<optype>::node_type;

    using boolnode = typename tree<bool>::node_type;

    using size_type = typename backend::size_type;

public:
    static inline size_type contraction_buffer_size(const hnode& A, bool use_capacity = false)
    {
        size_type maxdim = 0;
        for(size_type mode = 0; mode < A().nmodes(); ++mode)
        {
            auto _A = A().as_rank_3(mode, use_capacity);
            size_type dim = _A.shape(0)*_A.shape(1)*_A.shape(1);
            if(dim > maxdim){maxdim = dim;}
        }
        return maxdim;
    }

    
    static inline void evaluate(const hnode& A, triad& HA, triad& temp, triad& temp2, opnode& h)
    {
        try
        {
            //we only need to update the mean field Hamiltonian if it isn't the root node.
            if(!h.is_root())
            {
                for(size_type i=0; i < HA.size(); ++i)
                {
                    HA[i].resize(A().size(0), A().size(1));
                    temp[i].resize(A().size(0), A().size(1));
                    temp2[i].resize(A().size(0), A().size(1));
                }
                //resize the matrices in the event that the tensor objects have changed size
                //CALL_AND_HANDLE(h.resize_matrices(A.size(1), A.size(1)), "Failed to resize the single-particle Hamiltonian operator matrices.");

                size_type mode = h.child_id();
                const auto& h_p = h.parent();

                //#pragma omp parallel for default(shared) schedule(dynamic, 1)
                for(size_type ind = 0; ind < h().nterms(); ++ind)
                {
                    size_type ti = omp_get_thread_num();

                    //if the mean field operator is the identity then we don't need to do anything.
                    if(!h()[ind].is_identity_mf())
                    {
                        if(h()[ind].nmf_terms() == 1)
                        {
                            size_type pi = h()[ind].mf_indexing()[0].parent_index();

                            if(!h_p()[pi].is_identity_mf())
                            {
                                CALL_AND_HANDLE(kron_prod(h, ind, 0, A(), HA[ti], temp[ti]), "Failed to evaluate action of kronecker product operator.");
                                CALL_AND_HANDLE(HA[ti] = temp[ti]*trans(h_p()[pi].mf()), "Failed to apply action of parent mean field operator.");
                            }
                            else
                            {
                                CALL_AND_HANDLE(kron_prod(h, ind, 0, A(), temp[ti], HA[ti]), "Failed to evaluate action of kronecker product operator.");
                            }
                            CALL_AND_HANDLE(temp[ti] = conj(A().as_matrix()), "Failed to compute conjugate of the A matrix.");
                            try
                            {
                                auto _A = A().as_rank_3(mode);
                                auto _HA = HA[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                                auto _temp = temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                                CALL_AND_HANDLE(h()[ind].mf() = (contract(_temp, 0, 2, _HA, 0, 2).bind_workspace(temp2[ti])), "Failed when evaluating the final contraction.");
                            }                                         
                            catch(const std::exception& ex)
                            {
                                std::cerr << ex.what() << std::endl;
                                RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                            }   
                        }
                        else
                        {
                            h()[ind].mf().fill_zeros();
                            for(size_type it = 0; it < h()[ind].nmf_terms(); ++it)
                            {
                                size_type pi = h()[ind].mf_indexing()[it].parent_index();

                                if(!h_p()[pi].is_identity_mf())
                                {
                                    CALL_AND_HANDLE(kron_prod(h, ind, it, A(), HA[ti], temp[ti]), "Failed to evaluate action of kronecker product operator.");
                                    CALL_AND_HANDLE(HA[ti] = temp[ti]*trans(h_p()[pi].mf()), "Failed to apply action of parent mean field operator.");
                                }
                                else
                                {
                                    CALL_AND_HANDLE(kron_prod(h, ind, it, A(), temp[ti], HA[ti]), "Failed to evaluate action of kronecker product operator.");
                                }

                                CALL_AND_HANDLE(temp[ti] = conj(A().as_matrix()), "Failed to compute conjugate of the A matrix.");

                                try
                                {
                                    auto _A = A().as_rank_3(mode);
                                    auto _HA = HA[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                                    auto _temp = temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                                    CALL_AND_HANDLE(h()[ind].mf() += h()[ind].accum_coeff(it)*(contract(_temp, 0, 2, _HA, 0, 2).bind_workspace(temp2[ti])), "Failed when evaluating the final contraction.");
                                }                                         
                                catch(const std::exception& ex)
                                {
                                    std::cerr << ex.what() << std::endl;
                                    RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                                }                       
                            }
                        }
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate mean field operator at a node.");
        }
    }

public:
    static void kron_prod(const opnode& op, size_type ind, size_type ri, const hdata& A, mat& temp, mat& res)
    {
        try
        {
            bool first_call = true;
            bool res_set = true;

            const auto& spinds = op()[ind].mf_indexing()[ri].sibling_indices();
            for(size_type ni=0; ni<spinds.size(); ++ni)
            {
                size_type nu = spinds[ni][0];
                size_type cri = spinds[ni][1];

                auto _A = A.as_rank_3(nu);
                auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                //std::cerr << "A" << A.as_matrix() << std::endl;
                //std::cerr << "res" << res << std::endl;
                //std::cerr << "temp" << temp << std::endl;
        
                //std::cerr << op.parent()[nu]()[cri].spf() << std::endl;
                if(first_call)
                {     
                    CALL_AND_HANDLE(_res  = contract(op.parent()[nu]()[cri].spf(), 1, _A, 1), "Failed to compute kronecker product contraction.");      
                    res_set = true; first_call = false;
                }
                else if(res_set)
                {   
                    CALL_AND_HANDLE(_temp = contract(op.parent()[nu]()[cri].spf(), 1, _res, 1), "Failed to compute kronecker product contraction.");    
                    res_set = false;
                }
                else
                {               
                    CALL_AND_HANDLE(_res  = contract(op.parent()[nu]()[cri].spf(), 1, _temp, 1), "Failed to compute kronecker product contraction.");   
                    res_set = true;
                }
            }
            if(first_call){res_set = true;  res = A.as_matrix();}
            if(!res_set){res.swap_buffer(temp);}
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply kronecker product operator.");
        }
    }
};  //class mean field operator engine

}   //namespace ttns

#endif  //HTUCKER_MEAN_FIELD_OPERATOR_CORE_HPP//

