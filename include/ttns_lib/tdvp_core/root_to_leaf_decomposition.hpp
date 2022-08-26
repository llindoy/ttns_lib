#ifndef HTUCKER_ROOT_TO_LEAF_DECOMPOSITION_HPP
#define HTUCKER_ROOT_TO_LEAF_DECOMPOSITION_HPP

#include <linalg/linalg.hpp>

namespace ttns
{

template <typename T, typename backend> 
class root_to_leaf_decomposition_engine
{
    using hnode = httensor_node<T, backend>;
    using hdata = httensor_node_data<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using triad = std::vector<mat>;
    using matnode = typename tree<mat>::node_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using optype = operator_node_data<T, backend>;
    using opnode = typename tree<optype>::node_type;


    using size_type = typename hnode::size_type;
public:
    //helper functions for determining the size of temporary buffers needed for the decomposition engine
    static inline std::array<size_type, 2> maximum_matrix_dimension_node(const hnode& a, bool use_capacity = false)
    {
        std::array<size_type, 2> ret{{use_capacity ? a().capacity() : a().size(), 1}};
        size_type max_dim = a().hrank(use_capacity);
        for(size_type mode=0; mode < a().nmodes(); ++mode)
        {
            if(a().dim(mode, use_capacity) > max_dim){max_dim = a().dim(mode, use_capacity);}
        }
        ret[0] /= max_dim;  ret[1] *= max_dim;
        return ret;
    }

    template <typename engine>
    static inline size_type maximum_work_size_node(engine& eng, const hnode& a, mat& u, matnode& r, bool use_capacity = false)
    {
        try
        {
            size_type max_work_size;
            const hdata& A = a();       mat& U = u;        mat& R = r();
            
            //determine the maximum dimension of the hierarchical tucker decomposition object
            size_type max_dim = A.hrank(use_capacity);
            if(a.nmodes()>1)
            {
                for(size_type mode=0; mode < A.nmodes(); ++mode)
                {
                    if(A.dim(mode) > max_dim){max_dim = A.dim(mode, use_capacity);}
                }
            }

            //check that the result arrays have the correct capacity so that we can make sure we get the correct result
            if(use_capacity)
            {
                ASSERT(A.capacity() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
            }
            else            
            {
                ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
            }
            ASSERT(max_dim*max_dim <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");
            
            CALL_AND_HANDLE(U.resize(A.dimen(use_capacity), A.hrank(use_capacity)), "Failed when resizing the U matrix so that it has the correct shape.");
            CALL_AND_HANDLE(R.resize(A.hrank(use_capacity), A.hrank(use_capacity)), "Failed when resizing the R matrix so that it has the correct shape.");

            //first check the default ordering decomposition

            CALL_AND_HANDLE(max_work_size = eng.query_work_size(A.as_rank_2(use_capacity), u, r()), "Failed to query work size of the decomposition engine.");

            //now we test all of the possible reordering sizes 
            if(A.nmodes() != 1)
            {
                for(size_type mode=0; mode<A.nmodes(); ++mode)
                {
                    int d1 = A.hrank(use_capacity)*A.dimen(use_capacity)/A.dim(mode, use_capacity);     int d2 = A.dim(mode, use_capacity);
                    auto Ar = A.as_rank_2(use_capacity).reinterpret_shape(d1, d2);
                    auto Ur = U.reinterpret_shape(d1, d2);

                    CALL_AND_HANDLE(R.resize(d2, d2), "Failed when resizing the R matrix so that it has the correct shape.");

                    size_type work_size;
                    CALL_AND_HANDLE(work_size = eng.query_work_size(Ar, Ur, R), "Failed to query work size of the decomposition engine.");

                    if(work_size > max_work_size){max_work_size = work_size;}
                }
            }
            //and return the maximum worksize
            return max_work_size;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate the work size required for computing the root to leaf decomposition.");
        }
    }

    //function for resizing the node objects for the root to leaf decomposition
    static inline void resize_r_matrix(const hdata& a, mat& r, bool use_capacity = false)
    {
        size_type max_dim = a.hrank(use_capacity);
        if(a.nmodes()>1)
        {
            for(size_type mode = 0; mode<a.nmodes(); ++mode){max_dim = max_dim < a.dim(mode, use_capacity) ? a.dim(mode, use_capacity) : max_dim;}
        }
        r.resize(max_dim,max_dim);
    }

    //evaluate the root to leaf decomposition of a node for a given logical index.  Given the node tensor A and a mode index mode, returns the two matrices U, R
    template <typename engine>
    static inline void evaluate(engine& eng, const hnode& a, mat& u, matnode& r, mat& tm, size_type mode)
    {
        try
        {
            ASSERT(!a.is_leaf(), "Cannot evaluate root to leaf decomposition on a leaf node.")
            const hdata& A = a();     mat& U = u;        mat& R = r();        
            ASSERT(mode < A.nmodes(), "Failed to perform the root to leaf decomposition.  The node to be decomposed does not have the requested mode.");

            size_type d1 = (A.hrank()*A.dimen())/A.dim(mode);     size_type d2 = A.dim(mode);

            ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
            ASSERT(A.size() <= tm.capacity(), "The U matrix is not the same size as the input matrix."); 
            ASSERT(d2*d2 <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");

            CALL_AND_HANDLE(U.resize(A.size(0), A.size(1)), "Failed when resizing the U matrix so that it has the correct shape.");
            CALL_AND_HANDLE(tm.resize(d1, d2), "Failed when resizing the U matrix so that it has the correct shape.");
            CALL_AND_HANDLE(R.resize(d2, d2), "Failed when resizing the R matrix so that it has the correct shape.");

            //indices useful for reordering the A tensor
            size_type id1 = 1;                  size_type id2 = A.hrank()*A.dimen();
            for(size_type i=0; i<=mode; ++i){id1 *= A.dim(i);}
            id2 /= id1;

            //First we create a reordering of the matrix such that we have the index of interest as the fastest index of the matrix.
            //This is done by performing the operations:
            //  
            //
            CALL_AND_HANDLE(
            {
                //first we reinterpret 
                auto atens = A.as_matrix().reinterpret_shape(id1, id2);
                auto utens = U.reinterpret_shape(id2, id1);

                //now we permute the dimensions of this reinterpreted shape rank 3 tensor so that the middle dimension becomes the last dimension.
                utens = trans(atens);
            }
            , "Failed to unpack the matricisation back into the full rank tensor.");

            //now we reinterpret the reordered tensors as the correctly sized matrix and perform the singular values decomposition
            CALL_AND_HANDLE(
            {
                auto umat = U.reinterpret_shape(d1, d2);
                eng(umat, tm, R);
            }
            , "Failed when evaluating the decomposition of the resultant matricisation.");

            //now it is necessary to undo the permutation done before so that we can store the transformed U tensor
            CALL_AND_HANDLE(
            {
                auto utens = U.reinterpret_shape(id1, id2);
                auto ttens = tm.reinterpret_shape(id2, id1);

                utens = trans(ttens);
            }
            ,"Failed when repacking the matricisation.");
            
            //Finally we need to transpose the R matrix as we want its transpose for the remainder of the code.
            CALL_AND_HANDLE(eng.transposeV(R), "Failed to compute the inplace transpose of the R matrix.");
            
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating root to leaf decomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate root to leaf decomposition.");
        }
    }

    //applies the result of a nodes root to leaf decomposition at the current node.  This function sets A = U
    //We might want to combine this withe the evaluate routine and implement this with a swap operation to avoid the copy operation.
    static inline void apply_to_node(hnode& a, const mat& u)
    {
        try
        {
            ASSERT(a().as_matrix().shape() == u.shape(), "The U matrix and hierarchical tucker tensor node matrix are not the same size.");
            CALL_AND_HANDLE(a().as_matrix() = u, "Failed to set the value of the hierarchial tucker tensor node matrix.");
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying root to leaf decomposition at the current node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply root to leaf decomposition at the current node.");
        }
    }

    //applies the result of the root to leaf decomposition of a nodes parent to the node.  
    //we might want to implement this with a swap operation to avoid the copy operation
    static inline void apply_from_parent(hnode& a, const matnode& r, mat& tm)
    {
        try
        {
            ASSERT(!r.is_root(), "The apply to parent operation cannot act on the root node.");
            mat& A = a().as_matrix();   const mat& pr = r.parent()();

            ASSERT(A.size() <= tm.capacity(), "The temporary working matrix is not large enough to store the result.");
            ASSERT(pr.shape(0) == pr.shape(1) && pr.shape(0) == A.shape(1), "The parent's R matrix is not the correct size.");
            CALL_AND_HANDLE(tm.resize(A.shape()), "Failed to resize the temporary working matrix so that it has the correct shape.");

            CALL_AND_HANDLE(tm = A*pr, "Failed to evaluate contraction with parents R matrix.");
            CALL_AND_HANDLE(A = tm, "Failed to set the value of the hierarchical tucker tensor node matrix.");
            
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying root to leaf decomposition to parent of current node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply result of root to leaf decomposition to parent of current node.");
        }
    }



public:
    template <typename engine>
    static inline void evaluate(engine& eng, const hnode& a, mat& u, matnode& r, matnode& invmat, mat& tm, size_type mode, real_type eps)
    {
        try
        {
            ASSERT(!a.is_leaf(), "Cannot evaluate root to leaf decomposition on a leaf node.")
            const hdata& A = a();     mat& U = u;        mat& R = r();        mat& inv = invmat();
            ASSERT(mode < A.nmodes(), "Failed to perform the root to leaf decomposition.  The node to be decomposed does not have the requested mode.");

            size_type d1 = (A.hrank()*A.dimen())/A.dim(mode);     size_type d2 = A.dim(mode);

            ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
            ASSERT(A.size() <= tm.capacity(), "The U matrix is not the same size as the input matrix."); 
            ASSERT(d2*d2 <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");
            ASSERT(inv.shape(0) == d2 && inv.shape(1) == d2, "The regularised inverse R is not the correct size.");

            CALL_AND_HANDLE(U.resize(A.size(0), A.size(1)), "Failed when resizing the U matrix so that it has the correct shape.");
            CALL_AND_HANDLE(tm.resize(d1, d2), "Failed when resizing the U matrix so that it has the correct shape.");
            CALL_AND_HANDLE(R.resize(d2, d2), "Failed when resizing the R matrix so that it has the correct shape.");

            //indices useful for reordering the A tensor
            size_type id1 = 1;                  size_type id2 = A.hrank()*A.dimen();
            for(size_type i=0; i<=mode; ++i){id1 *= A.dim(i);}
            id2 /= id1;

            //First we create a reordering of the matrix such that we have the index of interest as the fastest index of the matrix.
            //This is done by performing the operations:
            //  
            //
            CALL_AND_HANDLE(
            {
                //first we reinterpret 
                auto atens = A.as_matrix().reinterpret_shape(id1, id2);
                auto utens = U.reinterpret_shape(id2, id1);

                //now we permute the dimensions of this reinterpreted shape rank 3 tensor so that the middle dimension becomes the last dimension.
                utens = trans(atens);
            }
            , "Failed to unpack the matricisation back into the full rank tensor.");

            //now we reinterpret the reordered tensors as the correctly sized matrix and perform the singular values decomposition
            CALL_AND_HANDLE(
            {
                auto umat = U.reinterpret_shape(d1, d2);
                eng.compute_including_regularised_inverse(umat, tm, R, inv, eps);
            }
            , "Failed when evaluating the decomposition of the resultant matricisation.");

            //now it is necessary to undo the permutation done before so that we can store the transformed U tensor
            CALL_AND_HANDLE(
            {
                auto utens = U.reinterpret_shape(id1, id2);
                auto ttens = tm.reinterpret_shape(id2, id1);

                utens = trans(ttens);
            }
            ,"Failed when repacking the matricisation.");
            
            //Finally we need to transpose the R matrix as we want its transpose for the remainder of the code.
            CALL_AND_HANDLE(eng.transposeV(R), "Failed to compute the inplace transpose of the R matrix.");
            
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating root to leaf decomposition.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate root to leaf decomposition.");
        }
    }

    //applies the result of the root to leaf decomposition of a nodes parent to the node.  
    //we might want to implement this with a swap operation to avoid the copy operation
    static inline void apply_from_parent(const hnode& a, const matnode& r, hnode& atilde)
    {
        try
        {
            ASSERT(!r.is_root(), "The apply to parent operation cannot act on the root node.");
            const mat& A = a().as_matrix();   mat& Atilde = atilde();  const mat& pr = r.parent()();

            ASSERT(A.size() == Atilde.size(), "The temporary working matrix is not large enough to store the result.");
            ASSERT(pr.shape(0) == pr.shape(1) && pr.shape(0) == A.shape(1), "The parent's R matrix is not the correct size.");

            CALL_AND_HANDLE(Atilde = A*pr, "Failed to evaluate contraction with parents R matrix.");
            
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying root to leaf decomposition to parent of current node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply result of root to leaf decomposition to parent of current node.");
        }
    }


};  //class root_to_leaf_decomposition

}   //namespace ttns

#endif  //HTUCKER_ROOT_TO_LEAF_DECOMPOSITION_HPP

