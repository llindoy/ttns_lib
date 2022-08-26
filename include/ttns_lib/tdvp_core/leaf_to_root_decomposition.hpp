#ifndef HTUCKER_LEAF_TO_ROOT_DECOMPOSITION_ENGINE_HPP
#define HTUCKER_LEAF_TO_ROOT_DECOMPOSITION_ENGINE_HPP

#include <linalg/linalg.hpp>

namespace ttns
{

template <typename T, typename backend> 
class leaf_to_root_decomposition_engine
{
    using hnode = httensor_node<T, backend>;
    using hdata = httensor_node_data<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using matnode = typename tree<mat>::node_type;

    using size_type = typename hnode::size_type;
public:
    //helper functions for determining the size of temporary buffers needed for the decomposition engine
    static inline std::array<size_type, 2> maximum_matrix_dimension_node(const hnode& a, bool use_capacity = false)
    {
        std::array<size_type, 2> ret{{a().dimen(use_capacity), a().hrank(use_capacity)}};
        return ret;
    }

    template <typename engine>
    static inline size_type maximum_work_size_node(engine& eng, const hnode& a, mat u, matnode& r, bool use_capacity = false)
    {
        try
        {
            const hdata& A = a();     mat& U = u;        mat& R = r();
            
            if(use_capacity){ASSERT(A.capacity() <= U.capacity(), "The U matrix is not the same size as the input matrix."); }
            else{ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); }
            ASSERT(A.hrank(use_capacity)*A.hrank(use_capacity) <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");
            
            CALL_AND_HANDLE(U.resize(A.dimen(use_capacity), A.hrank(use_capacity)), "Failed when resizing the U matrix so that it has the correct shape.");
            CALL_AND_HANDLE(R.resize(A.hrank(use_capacity), A.hrank(use_capacity)), "Failed when resizing the R matrix so that it has the correct shape.");
            CALL_AND_HANDLE(return eng.query_work_size(A.as_rank_2(use_capacity), U, R), "Failed to query work size of the decomposition engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate the work size required for computing the leaf to root decomposition.");
        }
    }
    
    //function for resizing the node objects for the leaf to root decomposition
    static inline void resize_r_matrix(const hdata& a, mat& r, bool use_capacity = false)
    {
        CALL_AND_RETHROW(r.resize(a.hrank(use_capacity), a.hrank(use_capacity)));
    }

    //evluate the leaf to root decomposition at a node.  Given the node tensor A returns the two matrices U, R such that
    //A = U R
    template <typename engine>
    static inline void evaluate(engine& eng, const hnode& a, mat& u, matnode& r)
    {
        try
        {
            const mat& A = a().as_matrix();     mat& U = u;        mat& R = r();
            ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
            ASSERT(a.hrank()*a.hrank() <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");
            
            CALL_AND_HANDLE(U.resize(A.shape(0), A.shape(1)), "Failed when resizing the U matrix so that it has the correct shape.");
            CALL_AND_HANDLE(R.resize(A.shape(1), A.shape(1)), "Failed when resizing the R matrix so that it has the correct shape.");
            CALL_AND_HANDLE(eng(A, U, R), "Failed when using the decomposition engine to evaluate the decomposition.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating leaf to root decomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate leaf to root decomposition.");
        }
    }

    //applies the result of a nodes leaf to root decomposition at the current node.  This function sets A = U
    static inline void apply_to_node(hnode& a, const mat& u)
    {
        try
        {
            ASSERT(a().as_matrix().shape() == u.shape(), "The U matrix and hierarchical tucker tensor node matrix are not the same size.");

            //we might want to change this for a swap operation rather than a copy as it will slightly reduce the numerical cost
            CALL_AND_HANDLE(a().as_matrix() = u, "Failed to set the value of the hierarchial tucker tensor node matrix.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying leaf to root decomposition at the current node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply leaf to root decomposition at the current node.");
        }
    }

    //applies the result of a nodes leaf to root decomposition to it's parent node.  This function computes the tensor X^{n}
    //obtained as A^{n-1}_I;j A^{n}_kji,l = U^{n-1}_I;j (R_{jj'} A^{n}_kj'i, l) = U^{n-1}_I;j X^{n}_kji,l
    static inline void apply_to_parent(hnode& a, const matnode& r, mat& pt)
    {
        try
        {
            ASSERT(!a.is_root(), "The apply to parent operation cannot act on the root node.");
            hdata& pa = a.parent()();   const mat& R = r();     pt.resize(pa.shape(0), pa.shape(1));

            ASSERT(pa.as_matrix().size() <= pt.capacity(), "The temporary working matrix is not large enough to store the result.");
            CALL_AND_HANDLE(pt.resize(pa.as_matrix().shape()), "Failed to resize the temporary working matrix so that it has the correct shape.");


            size_type mode = a.child_id();

            ASSERT(mode < pa.nmodes(), "The hierarchical tucker tensor is ill-formed.");

            auto pa_3 = pa.as_rank_3(mode);
            auto pt_3 = pt.reinterpret_shape(pa_3.shape());
            CALL_AND_HANDLE(pt_3 = contract(R, 1, pa_3, 1), "Failed to compute the requested contraction between the R matrix and the parent A tensor.");

            //we might want to change this for a swap operation rather than a copy as it will slightly reduce the numerical cost
            CALL_AND_HANDLE(pa.as_matrix() = pt, "Failed to copy temporary working matrix into hierarchical tucker tensor parent node matrix.");
            pt.resize(a().shape(0), a().shape(1));
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying result of leaf to root decomposition to parent of current node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply result of leaf to root decomposition to parent of current node.");
        }
    }

    //this function applies the transformation to both the present node and it's parent.
    static inline void apply(hnode& a, const mat& u, const matnode& r, mat& temp)
    {
        try
        {
            apply_to_node(a, u);
            apply_to_parent(a, r, temp);
        }        
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying the leaf to root transformation for a given node to the hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the leaf to root transformation for a given node to the hierarchical tucker tensor.");
        }
    }

};  //class leaf_to_root_decomposition

}   //namespace ttns

#endif  //HTUCKER_LEAF_TO_ROOT_DECOMPOSITION_ENGINE_HPP

