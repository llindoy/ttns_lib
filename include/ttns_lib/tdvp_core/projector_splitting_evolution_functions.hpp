#ifndef HTUCKER_PROJECTOR_SPLITTING_EVOLUTION_FUNCTIONS_HPP
#define HTUCKER_PROJECTOR_SPLITTING_EVOLUTION_FUNCTIONS_HPP

namespace ttns
{
template <typename T, typename backend>
class full_hamiltonian_action
{
    using opdata_type = operator_node_data<T, backend>;
    using coef_type = linalg::tensor<T, 1, linalg::blas_backend>;
    using mat = linalg::matrix<T, backend>;
    using matnode_type = typename tree<mat>::node_type;
    using opnode_type = typename tree<opdata_type>::node_type;
    using size_type = typename backend::size_type;
public:
    template <typename vtype, typename mat_type, typename rtype>
    inline void operator()(const vtype& v, const opnode_type& h, mat_type& t1, rtype& res)
    {
        try
        {
            size_type n1 = t1.shape(0);  size_type n2 = t1.shape(1);
            for(size_type ind=0; ind < h().nterms(); ++ind)
            {
                if(h()[ind].is_identity_spf())
                {
                    if(h()[ind].is_identity_mf())
                    {
                        if(ind == 0){CALL_AND_HANDLE(res = h()[ind].coeff()*v, "Failed to apply identity contribution.");}
                        else{CALL_AND_HANDLE(res += h()[ind].coeff()*v, "Failed to apply identity contribution.");}
                    }
                    else
                    {
                        if(ind == 0){CALL_AND_HANDLE(res = h()[ind].coeff()*v*trans(h()[ind].mf()), "Failed to apply the mean field contribution matrix.");}
                        else{CALL_AND_HANDLE(res += h()[ind].coeff()*v*trans(h()[ind].mf()), "Failed to apply the mean field contribution matrix.");}
                    }
                }
                else
                {

                    if(h()[ind].is_identity_mf())
                    {
                        if(ind == 0){CALL_AND_HANDLE(res = h()[ind].coeff()*h()[ind].spf()*v, "Failed to apply the single particle contribution.");}
                        else{CALL_AND_HANDLE(res += h()[ind].coeff()*h()[ind].spf()*v, "Failed to apply the single particle contribution.");}
                    }
                    else
                    {
                        CALL_AND_HANDLE(t1 = h()[ind].spf()*v, "Failed to apply the single particle contribution.");
                        if(ind == 0){CALL_AND_HANDLE(res = h()[ind].coeff()*t1*trans(h()[ind].mf()), "Failed to apply the mean field contribution.");}
                        else{CALL_AND_HANDLE(res += h()[ind].coeff()*t1*trans(h()[ind].mf()), "Failed to apply the mean field contribution.");}
                    }
                }
            }
            t1.resize(n1, n2);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the action of the full Hamiltonian at a node.");
        }
    }

};  //class full_hamiltonian_action

template <typename T, typename backend>
class coefficient_evolution_leaf
{
    using opdata_type = operator_node_data<T, backend>;
    using opnode_type = typename tree<opdata_type>::node_type;
    using size_type = typename backend::size_type;
    using op_type = sop_operator<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using matnode_type = typename tree<mat>::node_type;
public:
    template <typename vtype, typename mat_type, typename rtype>
    inline void operator()(const vtype& v, const opnode_type& h, const op_type& hprim, mat_type& t1, mat_type& t2, rtype& res)
    {   
        try
        {
            size_type n1 = t1.shape(0);  size_type n2 = t1.shape(1);
            
            for(size_type ind=0; ind < h().nterms(); ++ind)
            {
                if(h()[ind].is_identity_spf())
                {
                    T coeff(0);
                    if(h()[ind].nspf_terms() > 1){for(size_type i=0; i  < h()[ind].nspf_terms(); ++i){coeff += h()[ind].accum_coeff(i);}}
                    else{coeff = h()[ind].coeff();}

                    if(h()[ind].is_identity_mf())
                    {
                        if(ind == 0){CALL_AND_HANDLE(res = coeff*v, "Failed to apply identity contribution.");}
                        else{CALL_AND_HANDLE(res += coeff*v, "Failed to apply identity contribution.");}
                    }
                    else
                    {
                        if(ind == 0){CALL_AND_HANDLE(res = coeff*v*trans(h()[ind].mf()), "Failed to apply the mean field contribution matrix.");}
                        else{CALL_AND_HANDLE(res += coeff*v*trans(h()[ind].mf()), "Failed to apply the mean field contribution matrix.");}
                    }
                }
                else
                {
                    if(h()[ind].is_identity_mf())
                    {
                        if(ind == 0)
                        {
                            {
                                T coeff = h()[ind].nspf_terms() > 1 ? h()[ind].accum_coeff(0) : h()[ind].coeff();
                                auto& indices = h()[ind].spf_indexing()[0][0];
                                CALL_AND_HANDLE(hprim(indices[0], indices[1]).apply(v, t2), "Failed to apply leaf operator.");
                                res = coeff*t2;
                            }

                            for(size_type i = 1; i < h()[ind].nspf_terms(); ++i)
                            {
                                T coeff = h()[ind].accum_coeff(i);
                                auto& indices = h()[ind].spf_indexing()[i][0];
                                CALL_AND_HANDLE(hprim(indices[0], indices[1]).apply(v, t2), "Failed to apply leaf operator.");
                                res += coeff*t2;
                            }
                        }
                        else
                        {
                            for(size_type i = 0; i < h()[ind].nspf_terms(); ++i)
                            {
                                T coeff = h()[ind].nspf_terms() > 1 ? h()[ind].accum_coeff(i) : h()[ind].coeff();
                                auto& indices = h()[ind].spf_indexing()[i][0];
                                CALL_AND_HANDLE(hprim(indices[0], indices[1]).apply(v, t2), "Failed to apply leaf operator.");
                                res += coeff*t2;
                            }
                        }
                    }
                    else
                    {
                        {
                            T coeff = h()[ind].nspf_terms() > 1 ? h()[ind].accum_coeff(0) : h()[ind].coeff();
                            auto& indices = h()[ind].spf_indexing()[0][0];
                            CALL_AND_HANDLE(hprim(indices[0], indices[1]).apply(v, t1), "Failed to apply leaf operator.");
                            t2 = coeff*t1;
                        }
                        for(size_type i = 1; i < h()[ind].nspf_terms(); ++i)
                        {
                            T coeff = h()[ind].accum_coeff(i);
                            auto& indices = h()[ind].spf_indexing()[i][0];
                            CALL_AND_HANDLE(hprim(indices[0], indices[1]).apply(v, t1), "Failed to apply leaf operator.");
                            t2 += coeff*t1;
                        }

                        if(ind == 0){CALL_AND_HANDLE(res = t2*trans(h()[ind].mf()), "Failed to apply the mean field contribution.");}
                        else{CALL_AND_HANDLE(res += t2*trans(h()[ind].mf()), "Failed to apply the mean field contribution.");}
                    }
                }
            }
            t1.resize(n1, n2);
            t2.resize(n1, n2);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the leaf coefficient evolution operator at a node.");
        }
    }
};  //class coefficient_evolution_leaf


template <typename T, typename backend>
class coefficient_evolution_branch
{
    using opdata_type = operator_node_data<T, backend>;
    using opnode_type = typename tree<opdata_type>::node_type;
    using coef_type = linalg::tensor<T, 1, linalg::blas_backend>;
    using size_type = typename backend::size_type;
    using dims_type = std::vector<size_type>;
    using mat = linalg::matrix<T, backend>;
    using matnode_type = typename tree<mat>::node_type;
public:

    template <typename vtype, typename mat_type, typename rtype>
    inline void operator()(const vtype& v, const opnode_type& h, size_type hrank, const dims_type& dims, mat_type& t1, mat_type& t2, mat_type& t3, rtype& res)
    {   
        try
        {
            size_type n1 = t1.shape(0);  size_type n2 = t1.shape(1);
            for(size_type ind=0; ind < h().nterms(); ++ind)
            {
                if(h()[ind].is_identity_spf())
                {
                    T coeff(0);
                    if(h()[ind].nspf_terms() > 1){for(size_type i=0; i  < h()[ind].nspf_terms(); ++i){coeff += h()[ind].accum_coeff(i);}}
                    else{coeff = h()[ind].coeff();}

                    if(h()[ind].is_identity_mf())
                    {
                        if(ind == 0){CALL_AND_HANDLE(res = coeff*v, "Failed to apply identity contribution.");}
                        else{CALL_AND_HANDLE(res += coeff*v, "Failed to apply identity contribution.");}
                    }
                    else
                    {
                        if(ind == 0){CALL_AND_HANDLE(res = coeff*v*trans(h()[ind].mf()), "Failed to apply the mean field contribution matrix.");}
                        else{CALL_AND_HANDLE(res += coeff*v*trans(h()[ind].mf()), "Failed to apply the mean field contribution matrix.");}
                    }
                }
                else
                {
                    using kpo = kronecker_product_operator<T, backend>;

                    
                    if(h()[ind].is_identity_mf())
                    {
                        if(ind == 0)
                        {
                            T coeff = h()[ind].nspf_terms() > 1 ? h()[ind].accum_coeff(0) : h()[ind].coeff();
                            CALL_AND_HANDLE(kpo::apply(h, ind, 0, hrank, dims, v, t1, t2), "Failed to apply kronecker product operator.");
                            res = coeff*t2;
                            for(size_type i = 1; i < h()[ind].nspf_terms(); ++i)
                            {
                                coeff = h()[ind].accum_coeff(i);
                                CALL_AND_HANDLE(kpo::apply(h, ind, i, hrank, dims, v, t1, t2), "Failed to apply kronecker product operator.");
                                res += coeff*t2;
                            }
                        }
                        else
                        {
                            for(size_type i = 0; i < h()[ind].nspf_terms(); ++i)
                            {
                                T coeff = h()[ind].nspf_terms() > 1 ? h()[ind].accum_coeff(i) : h()[ind].coeff();
                                CALL_AND_HANDLE(kpo::apply(h, ind, i, hrank, dims, v, t1, t2), "Failed to apply kronecker product operator.");
                                res += coeff*t2;
                            }
                        }
                    }
                    else
                    {
                        T coeff = h()[ind].nspf_terms() > 1 ? h()[ind].accum_coeff(0) : h()[ind].coeff();
                        CALL_AND_HANDLE(kpo::apply(h, ind, 0, hrank, dims, v, t1, t2), "Failed to apply kronecker product operator.");
                        t3 = coeff*t2;
                        for(size_type i = 1; i < h()[ind].nspf_terms(); ++i)
                        {
                            coeff = h()[ind].accum_coeff(i);
                            CALL_AND_HANDLE(kpo::apply(h, ind, i, hrank, dims, v, t1, t2), "Failed to apply kronecker product operator.");
                            t3 += coeff*t2;
                        }

                        if(ind == 0){CALL_AND_HANDLE(res = t3*trans(h()[ind].mf()), "Failed to apply the mean field contribution.");}
                        else{CALL_AND_HANDLE(res += t3*trans(h()[ind].mf()), "Failed to apply the mean field contribution.");}
                    }
                }
            }
            t1.resize(n1, n2);
            t2.resize(n1, n2);
            t3.resize(n1, n2);
        }        
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the branch coefficient evolution operator at a node.");
        }
    }
};  //class coefficient_evolution_branch

}   //namespace ttns

#endif  //HTUCKER_PROJECTOR_SPLITTING_EVOLUTION_FUNCTIONS_HPP//

