#ifndef HTUCKER_ADAPT_BOND_DIMENSION_HPP
#define HTUCKER_ADAPT_BOND_DIMENSION_HPP

#include <memory>
#include "../utils/zip.hpp"
#include "../httensor.hpp"
#include "../operators/sop_operator.hpp"
#include "../ttn_nodes/operator_node.hpp"


#include "projector_splitting_evolution_functions.hpp"
#include "decomposition_engine.hpp"
#include "projector_splitting_evolution_functions.hpp"
#include "operator_container.hpp"

namespace ttns
{

//in order to adapt the number of single particle functions we will use the local in time error to quantify the change in error upon adding
//the additional SPFs.  Applying the gauge condition that the SPFs and their derivatives are orthonormal we may constuct these new functions by evaluating the eigenvalues
//of \Gamma^{z_l} = \bra{\Phi_I^{z_l} \Psi_J^{z_l}}\hat{H}\ket{\Psi}\bra{\Psi}\hat{H}\ket{\Phi_J^{z_l} \Psi_J^{z_j}}.  It is straightforward to show that this is equivalent 
//to finding the singular vectors of h A H.  The order in which we attempt to add new functions is arbitrary but we note that for the completely singular 
template <typename T, typename backend = blas_backend>
class adapt_bond_dimension
{
public:
    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename httensor<T,backend>::size_type;
    using engine_type = decomposition_engine<T, backend, false>;
    using matrix_type = matrix<T, backend>;
    using l2r = leaf_to_root_decomposition_engine<T, backend>;
    
    using operator_type = sop_operator<T, backend>;
public:
    adapt_bond_dimension(const httensor<T, backend>& A,size_type max_spf, real_type tol) 
    {
        CALL_AND_HANDLE(resize(A), "Failed to resize adapt_bond_dimension object.");
    }
    adapt_bond_dimension(const adapt_bond_dimension& o) = default;
    adapt_bond_dimension(adapt_bond_dimension&& o) = default;

    adapt_bond_dimension& operator=(const adapt_bond_dimension& o) = default;
    adapt_bond_dimension& operator=(adapt_bond_dimension&& o) = default;

    void resize(const httensor<T, backend>& A)
    {
        try
        {
            size_type max_hrank = 0;
            size_type max_size = 0;
            for(const auto& a : A)
            {
                if(a().size() > max_size){max_size = a().size();}
                if(a().hrank() > max_hrank){max_hrank = a().hrank();}
            }
            CALL_AND_HANDLE(m_u.resize(1, max_size), "Failed to resize u tensor.");
            CALL_AND_HANDLE(m_temp.resize(1, max_size), "Failed to resize temporary tensor.");
            CALL_AND_HANDLE(m_HA.resize(1, max_size), "Failed to resize temporary tensor.");
            CALL_AND_HANDLE(m_r.resize(max_hrank, max_hrank), "Failed to resize r tensor.");

            try
            {
                m_ortho_engine.template resize<l2r>(A, m_u, m_r);
            }
            catch(const std::exception& ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to resize the decomposition engine object.");
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize adapt_bond_dimension object.");
        }
    }

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_u.clear(), "Failed to clear the u array.");
            CALL_AND_HANDLE(m_r.clear(), "Failed to clear the r array.");
            CALL_AND_HANDLE(m_temp.clear(), "Failed to clear the temporary working array.");
            CALL_AND_HANDLE(m_ortho_engine.clear(), "Failed to clear the orthogonalisation engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear adapt_bond_dimension object.");
        }
    }


    void operator()(httensor<T, backend>& A, const operator_type& op, operator_container<T, backend>& ham)
    {
        try
        {
            //iterate over the httensor in reverse order (so we access children before their parents)
            for(auto& a : reverse(A))
            {
                auto& a = std::get<0>(z);
                auto& r = m_r;
                auto& u = m_u;
                m_u.resize(a().shape(0), a().shape(1));
                m_r.resize(a().hrank(), a().hrank());
                m_temp.resize(a().shape(0), a().shape(1));

                //if this isn't the root node we start by evaluating the action of the Hamiltonian on the node
                if(!a.is_root())
                {
                    if(!a.is_leaf())
                    {
                        coefficient_evolution_branch<T, backend> ceb;   //used for evolving the coefficient tensors of branch nodes
                    }
                    else
                    {
                        coefficient_evolution_leaf<T, backend> cel;     //used for evolving the coefficient tensors of leaf nodes
                    }
                }
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("orthogonalising the hierarchical tucker tensor object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to orthogonalise the hierarchical tucker tensor object.");
        }
    }

protected:
    //temporary objects for storing the intermediate quantities required for computing the orthogonalisation of the hierarchical tucker tensor
    matrix_type m_u;
    matrix_type m_r;
    matrix_type m_temp;
    matrix_type m_HA;

    //an object wrapping the singular value decomposition.  This also stores the workspace arrays required for the computations.
    engine_type m_ortho_engine;
};

}   //namespace ttns

#endif //HTUCKER_ADAPT_BOND_DIMENSION_HPP//
