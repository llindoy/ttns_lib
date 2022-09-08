#ifndef HTUCKER_ROOT_TO_LEAF_DECOMPOSITION_ENGINE_HPP
#define HTUCKER_ROOT_TO_LEAF_DECOMPOSITION_ENGINE_HPP


#include <memory>
#include <zip.hpp>
#include "../tdvp_core/root_to_leaf_decomposition.hpp"
#include "../tdvp_core/decomposition_engine.hpp"
#include "../tdvp_core/tdvp_tree_traversal.hpp"

namespace ttns
{

//might make the type of orthogonalisation a template parameter (all for qr)
template <typename T, typename backend = blas_backend>
class root_to_leaf_decomposition
{
public:
    using real_type = typename tmp::get_real_type<T>::type;
    using size_type = typename httensor<T,backend>::size_type;
    using engine_type = decomposition_engine<T, backend>;
    using matrix_type = matrix<T, backend>;
    using r2l = root_to_leaf_decomposition_engine<T, backend>;

public:
    root_to_leaf_decomposition() {}
    root_to_leaf_decomposition(const httensor<T, backend>& A, bool use_capacity = false) 
    {
        CALL_AND_HANDLE(resize(A, use_capacity), "Failed to constructor root_to_leaf object.  Failed to resize buffers.");
    }
    root_to_leaf_decomposition(const root_to_leaf_decomposition<T, backend>& A) = default;
    root_to_leaf_decomposition(root_to_leaf_decomposition<T, backend>&& A) = default;

    root_to_leaf_decomposition& operator=(const root_to_leaf_decomposition<T,backend>& A) = default;
    root_to_leaf_decomposition& operator=(root_to_leaf_decomposition<T, backend>&& A) = default;

    void resize(const httensor<T, backend>& A, bool use_capacity = false)
    {
        try
        {
            CALL_AND_HANDLE(m_r.construct_topology(A), "Failed to construct the topology of the r tensor.");
            CALL_AND_HANDLE(m_traversal.resize(A), "Failed to initialise the tree traversal object.");

            size_type max_size = 0;
            using utils::zip;
            for(auto z : zip(m_r, A))
            {
                size_type asize = use_capacity ? a().capacity() : a().size();
                if(asize > max_size){max_size = asize;}
                auto& r = std::get<0>(z);   const auto& a = std::get<1>(z);
                CALL_AND_HANDLE(r2l::resize_r_matrix(a(), r(), use_capacity), "Failed to resize elements of the r tensor.");
            }
            CALL_AND_HANDLE(m_temp.resize(1, max_size), "Failed to resize temporary tensor.");
            CALL_AND_HANDLE(m_u.resize(1, maxsize), "Failed to resize u tensor.");
            try
            {
                m_ortho_engine.template resize<r2l>(A, m_u, m_r, use_capacity);
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
            RAISE_EXCEPTION("Failed to resize root_to_leaf object.");
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
            CALL_AND_HANDLE(m_traversal.clear(), "Failed to clear the traversal array.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear root_to_leaf object.");
        }
    }

    void operator()(httensor<T, backend>& A)
    {
        try
        {
            ASSERT(has_same_structure(A, m_u), "The input hierarchical tucker tensor does not have the same structure as the root_to_leaf object internal buffers.");

            m_traversal.reset_times_visited();

            for(size_type id : m_traversal) 
            {
                auto& a = A[id];   auto& u = m_u;   auto& r = m_r[id];
                m_temp.resize(a().shape(0), a().shape(1));
                m_u.resize(a().shape(0), a().shape(1));
                size_type mode = m_traversal.times_visited(id);
                if(mode < a.nmodes())
                {
                    //if it is the first time we are accessing the mode then we apply the parents decomposition to this node (provided it is not the root)
                    if(!a.is_root() && mode == 0)
                    {
                        CALL_AND_HANDLE(r2l::apply_from_parent(a, r, m_temp), "Failed to apply the parents root to leaf decomposition to the current node.");
                    }
    
                    //then provided it is not a leaf node we evaluate the decomposition of the node and apply it inplace
                    if(!a.is_leaf())
                    {   
                        CALL_AND_HANDLE(r2l::evaluate(m_ortho_engine, a, u, r, m_temp, mode), "Failed to compute the root to leaf decomposition for a node.");
                        CALL_AND_HANDLE(r2l::apply_to_node(a, u), "Failed to apply the result of the root to leaf decomposition to the current node.");
                    }
                }
                
                ++(m_traversal.times_visited(id));
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing root to leaf decomposition of a hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute root to leaf decomposition of a hierarchical tucker tensor.");
        }
    }

protected:
    //temporary objects for storing the intermediate quantities required for computing the root to leaf decomposition of the hierarchical tucker tensor
    matrix_type m_u;
    tree<matrix_type> m_r;
    matrix_type m_temp;

    //an object wrapping the decomposition engine.  This also stores the workspace arrays required for the computations.
    engine_type m_ortho_engine;

    //an object storing the traversal order required for evaluating the root to leaf decomposition
    tdvp_tree_traversal m_traversal;

};

}   //namespace ttns

#endif //HTUCKER_ROOT_TO_LEAF_DECOMPOSITION_ENGINE_HPP//
