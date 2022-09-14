#ifndef HTUCKER_MATRIX_ELEMENT_CORE_HPP
#define HTUCKER_MATRIX_ELEMENT_CORE_HPP

#include <linalg/linalg.hpp>
#include "kronecker_product_operator_helper.hpp"

namespace ttns
{


//A class containing static functions that evaluate the contractions required for computing matrix elements.
template <typename T, typename backend>
class matrix_element_engine
{
public:
    using hnode = httensor_node<T, backend>;
    using hdata = httensor_node_data<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using matnode = typename tree<mat>::node_type;
    using boolnode = typename tree<bool>::node_type;

    using size_type = typename hnode::size_type;
    using op_base = ops::primitive<T, backend>;

    //function for resizing the node objects for the leaf to root decomposition
    static inline void resize(const hdata& a, mat& r, bool use_capacity = false)
    {
        if(use_capacity)
        {
            CALL_AND_RETHROW(r.reallocate(a.hrank(use_capacity)*a.hrank(use_capacity)));
        }
        CALL_AND_RETHROW(r.resize(a.hrank(), a.hrank()));
    }

    static inline void resize(const hdata& a, const hdata& /* b */, mat& r, bool use_capacity = false)
    {
        if(use_capacity)
        {
            CALL_AND_RETHROW(r.reallocate(a.hrank(use_capacity)*a.hrank(use_capacity)));
        }
        CALL_AND_RETHROW(r.resize(a.hrank(), a.hrank()));
    }

    static inline void compute_leaf(const hnode& p, matnode& res, boolnode& is_identity)
    {
        CALL_AND_RETHROW(compute_leaf(p, res, is_identity, false));
    }

    static inline void compute_leaf(const hnode& p, matnode& res, boolnode& is_identity, bool compute_explicit)
    {
        try
        {
            ASSERT(p.is_leaf(), "The input node is not a leaf node.");
            CALL_AND_HANDLE(res().resize(p().hrank(), p().hrank()), "Failed to resize matel object.");
            if(! (res.size() == p.size() && res.size() == p.size()))
            {
                CALL_AND_HANDLE(res().resize(p().hrank(), p().hrank()), "Failed to resize matel object.");
            }

            if(!p().is_orthogonalised() || compute_explicit)
            {
                const auto& psi = p().as_matrix();
                CALL_AND_HANDLE(res() = adjoint(psi)*psi, "Failed to apply the leaf node contraction.");
            }
            is_identity() = p().is_orthogonalised();
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing leaf node norm squared of hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute leaf node norm squared of hierarchical tucker tensor.");
        }
    }



    static inline void compute_leaf(const hnode& b, const hnode& k, matnode& res, boolnode& is_identity)
    {
        try
        {
            ASSERT(b.is_leaf(), "The input bra node is not a leaf node.");
            ASSERT(k.is_leaf(), "The input ket node is not a leaf node.");
            if(&b == &k)
            {
                CALL_AND_HANDLE(compute_leaf(k, res, is_identity), "Failed to treat the case where the bra and ket vector correspond to the same vector.");
            }   
            else
            {
                ASSERT(b().dims() == k().dims() && b().hrank() == k().hrank(), "Inner products between bra and ket nodes are only supported for nodes of the same size.");

                if(! (res.size() == b.size() && res.size() == k.size()))
                {
                    CALL_AND_HANDLE(res().resize(b().hrank(), k().hrank()), "Failed to resize matel object.");
                }
            
                CALL_AND_HANDLE(res().resize(b().hrank(), k().hrank()), "Failed to resize matel object.");
                const auto& bra = b().as_matrix();      const auto& ket = k().as_matrix();
                CALL_AND_HANDLE(res() = adjoint(bra)*ket, "Failed to apply the leaf node contraction.");
                is_identity() = false;
            }
            
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing leaf node inner product between two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute leaf node inner product between two hierarchical tucker tensors.");
        }
    }
    
    template <typename op_type>
    static inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, void>::type compute_leaf(op_type& op, const hnode& p, mat& temp, matnode& res, boolnode& is_identity)
    {
        try
        {
            if(op.is_identity())
            {
                CALL_AND_HANDLE(compute_leaf(p, res, is_identity), "Failed to treate the case where the operator is the identity operator.");
            }
            else
            {
                ASSERT(p.is_leaf(), "The input node is not a leaf node.");
                ASSERT(op.size() == p().dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the httensor node.");

                if(! (res.size() == p.size() && res.size() == p.size()))
                {
                    CALL_AND_HANDLE(res().resize(p().hrank(), p().hrank()), "Failed to resize matel object.");
                }

                const auto& psi = p().as_matrix();      auto& HA = temp;
                CALL_AND_HANDLE(op.apply(psi, HA), "Failed to evaluate the action of the operator on the ket vector.");
                CALL_AND_HANDLE(res() = adjoint(psi)*HA, "Failed to apply the leaf node contraction.");
                is_identity() = false;
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
    }

    static inline void compute_leaf(std::shared_ptr<op_base> op, const hnode& p, mat& temp, matnode& res, boolnode& is_identity)
    {
        try
        {
            if(op->is_identity())
            {
                CALL_AND_HANDLE(compute_leaf(p, res, is_identity), "Failed to treate the case where the operator is the identity operator.");
            }
            else
            {
                ASSERT(p.is_leaf(), "The input node is not a leaf node.");
                ASSERT(op->size() == p().dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the httensor node.");

                if(! (res.size() == p.size() && res.size() == p.size()))
                {
                    CALL_AND_HANDLE(res().resize(p().hrank(), p().hrank()), "Failed to resize matel object.");
                }

                const auto& psi = p().as_matrix();      auto& HA = temp;
                CALL_AND_HANDLE(op->apply(psi, HA), "Failed to evaluate the action of the operator on the ket vector.");
                CALL_AND_HANDLE(res() = adjoint(psi)*HA, "Failed to apply the leaf node contraction.");
                is_identity() = false;
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
    }

    template <typename op_type>
    static inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, void>::type compute_leaf(op_type& op, const hnode& b, const hnode& k, mat& temp, matnode& res, boolnode& is_identity)
    {
        try
        {
            if(op.is_identity())
            {
                CALL_AND_HANDLE(compute_leaf(b, k, res, is_identity), "Failed to treat the case where the operator is the identity operator.");
            }
            else if(&b == &k)
            {
                CALL_AND_HANDLE(compute_leaf(op, k, temp, res, is_identity), "Failed to treat the case where the bra and ket httensor nodes are the same operator.");
            }
            else
            {
                ASSERT(b.is_leaf(), "The input bra node is not a leaf node.");
                ASSERT(k.is_leaf(), "The input ket node is not a leaf node.");

            
                ASSERT(b().dims() == k().dims() && b().hrank() == k().hrank(), "Inner products between bra and ket nodes are only supported for nodes of the same size.");
                ASSERT(op.size() == b().dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the httensor node.");

                if(! (res.size() == b.size() && res.size() == k.size()))
                {
                    CALL_AND_HANDLE(res().resize(b().hrank(), k().hrank()), "Failed to resize matel object.");
                }

                const auto& bra = b().as_matrix();      const auto& ket = k().as_matrix();      auto& HA = temp;
                CALL_AND_HANDLE(op.apply(ket, HA), "Failed to evaluate the action of the operator on the ket vector.");
                CALL_AND_HANDLE(res() = adjoint(bra)*HA, "Failed to apply the leaf node contraction.");
            
                is_identity() = false;
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing matrix element of an operator between the leaf nodes of two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute a matrix element of an operator between the leaf nodes of two hierarchical tucker tensors.");
        }
    }

    static inline void compute_leaf(std::shared_ptr<op_base> op, const hnode& b, const hnode& k, mat& temp, matnode& res, boolnode& is_identity)
    {
        try
        {
            if(op->is_identity())
            {
                CALL_AND_HANDLE(compute_leaf(b, k, res, is_identity), "Failed to treat the case where the operator is the identity operator.");
            }
            else if(&b == &k)
            {
                CALL_AND_HANDLE(compute_leaf(op, k, temp, res, is_identity), "Failed to treat the case where the bra and ket httensor nodes are the same operator.");
            }
            else
            {
                ASSERT(b.is_leaf(), "The input bra node is not a leaf node.");
                ASSERT(k.is_leaf(), "The input ket node is not a leaf node.");
                ASSERT(b().dims() == k().dims() && b().hrank() == k().hrank(), "Inner products between bra and ket nodes are only supported for nodes of the same size.");
                ASSERT(op->size() == b().dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the httensor node.");

                if(! (res.size() == b.size() && res.size() == k.size()))
                {
                    CALL_AND_HANDLE(res().resize(b().hrank(), k().hrank()), "Failed to resize matel object.");
                }

                const auto& bra = b().as_matrix();      const auto& ket = k().as_matrix();      auto& HA = temp;
                CALL_AND_HANDLE(op->apply(ket, HA), "Failed to evaluate the action of the operator on the ket vector.");
                CALL_AND_HANDLE(res() = adjoint(bra)*HA, "Failed to apply the leaf node contraction.");
            
                is_identity() = false;
            }
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing matrix element of an operator between the leaf nodes of two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute a matrix element of an operator between the leaf nodes of two hierarchical tucker tensors.");
        }
    }

    static inline void compute_branch(const hnode& p, mat& HA, mat& temp, matnode& res, boolnode& is_identity)
    {
        CALL_AND_RETHROW(compute_branch(p, HA, temp, res, is_identity, false));
    }

    static inline void compute_branch(const hnode& p, mat& HA, mat& temp, matnode& res, boolnode& is_identity, bool compute_explicit)
    {
        try
        {
            ASSERT(!p.is_leaf() && !res.is_leaf(), "Cannot apply branch contraction to a leaf node.");
            if(! (res.size() == p.size() && res.size() == p.size()))
            {
                CALL_AND_HANDLE(res().resize(p().hrank(), p().hrank()), "Failed to resize matel object.");
            }

            //check the size of the child nodes and check if all child nodes are identity operators
            bool all_idop = true; 
            for(size_type i=0; i<is_identity.size(); ++i)
            {
                ASSERT(res[i]().size(0) == res[i]().size(1) && res[i]().size(1) == p.dim(i), "The child operator nodes are not the correct shape.");
                all_idop = all_idop && is_identity[i]();
            }

            //if some of the children are not the identity operators then we need to evaluate the kronecker product operator.
            if(!all_idop || compute_explicit)
            {
                const auto& psi = p().as_matrix();  auto& ha = HA;    auto& t = temp;
    
                using kpo = kronecker_product_operator<T, backend>;
                CALL_AND_HANDLE(kpo::apply(res, is_identity, p(), t, ha), "Failed to apply kronecker product operator.");
                CALL_AND_HANDLE(res() = adjoint(psi)*ha, "Failed to apply matrix product to obtain result.");
                is_identity() = false;
            }
            //if all the children are identity operators then we don't need to work out their action on the current hierarhical tucker tensor node
            else
            {
                //if p is not orthogonalised then we still need to do some calculation
                if(!p().is_orthogonalised() || p.is_root())
                {
                    const auto& psi = p().as_matrix();
                    CALL_AND_HANDLE(res() = adjoint(psi)*psi, "Failed to apply the branch node contraction.");
                }
                is_identity() = p().is_orthogonalised();
            }
        }        
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
    }

    static inline void compute_branch(const hnode& b, const hnode& k, mat& HA, mat& temp, matnode& res, boolnode& is_identity)
    {
        try
        {
            if(&b == &k)
            {
                CALL_AND_HANDLE(compute_branch(k, HA, temp, res, is_identity), "Failed to treat the case where the bra and ket httensor nodes are the same operator.");
            }
            else
            {
                ASSERT(!b.is_leaf() && !k.is_leaf() && !res.is_leaf(), "Cannot apply branch contraction to a leaf node.");
                if(! (res.size() == b.size() && res.size() == k.size()))
                {
                    CALL_AND_HANDLE(res().resize(b().hrank(), k().hrank()), "Failed to resize matel object.");
                }

                //check the size of the child nodes and check if all child nodes are identity operators
                bool all_idop = true; 
                for(size_type i=0; i<is_identity.size(); ++i)
                {
                    ASSERT(res[i]().size(0) == res[i]().size(1) && res[i]().size(1) == k.dim(i), "The child operator nodes are not the correct shape.");
                    all_idop = all_idop && is_identity[i]();
                }

                //if some of the children are not the identity operators then we need to evaluate the kronecker product operator.
                if(!all_idop)
                {
                    const auto& bra = b().as_matrix();  auto& ha = HA;    auto& t = temp;
    
                    using kpo = kronecker_product_operator<T, backend>;
                    CALL_AND_HANDLE(kpo::apply(res, is_identity, k(), t, ha), "Failed to apply kronecker product operator.");
                    CALL_AND_HANDLE(res() = adjoint(bra)*ha, "Failed to apply matrix product to obtain result.");
                    is_identity() = false;
                }
                //if all the children are identity operators then we don't need to work out their action on the current hierarhical tucker tensor node
                else
                {
                    const auto& bra = b().as_matrix();  const auto& ket = k().as_matrix();
                    CALL_AND_HANDLE(res() = adjoint(bra)*ket, "Failed to apply the branch node contraction.");
                    is_identity() = false;
                }
            }
        }        
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
    }


public:
    static inline void ct_leaf(const hnode& p, matnode& res, boolnode& is_identity, bool compute_explicit)
    {
        try
        {
            ASSERT(p.is_leaf(), "The input node is not a leaf node.");
            CALL_AND_HANDLE(res().resize(p().hrank(), p().hrank()), "Failed to resize matel object.");
            if(! (res.size() == p.size() && res.size() == p.size()))
            {
                CALL_AND_HANDLE(res().resize(p().hrank(), p().hrank()), "Failed to resize matel object.");
            }

            if(!p().is_orthogonalised() || compute_explicit)
            {
                const auto& psi = p().as_matrix();
                CALL_AND_HANDLE(res() = transpose(psi)*psi, "Failed to apply the leaf node contraction.");
            }
            is_identity() = p().is_orthogonalised();
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing leaf node norm squared of hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute leaf node norm squared of hierarchical tucker tensor.");
        }
    }

    static inline void ct_branch(const hnode& p, mat& HA, mat& temp, matnode& res, boolnode& is_identity, bool compute_explicit)
    {
        try
        {
            ASSERT(!p.is_leaf() && !res.is_leaf(), "Cannot apply branch contraction to a leaf node.");
            if(! (res.size() == p.size() && res.size() == p.size()))
            {
                CALL_AND_HANDLE(res().resize(p().hrank(), p().hrank()), "Failed to resize matel object.");
            }

            //check the size of the child nodes and check if all child nodes are identity operators
            bool all_idop = true; 
            for(size_type i=0; i<is_identity.size(); ++i)
            {
                ASSERT(res[i]().size(0) == res[i]().size(1) && res[i]().size(1) == p.dim(i), "The child operator nodes are not the correct shape.");
                all_idop = all_idop && is_identity[i]();
            }

            //if some of the children are not the identity operators then we need to evaluate the kronecker product operator.
            if(!all_idop || compute_explicit)
            {
                const auto& psi = p().as_matrix();  auto& ha = HA;    auto& t = temp;
    
                using kpo = kronecker_product_operator<T, backend>;
                CALL_AND_HANDLE(kpo::apply(res, is_identity, p(), t, ha), "Failed to apply kronecker product operator.");
                CALL_AND_HANDLE(res() = transpose(psi)*ha, "Failed to apply matrix product to obtain result.");
                is_identity() = false;
            }
            //if all the children are identity operators then we don't need to work out their action on the current hierarhical tucker tensor node
            else
            {
                //if p is not orthogonalised then we still need to do some calculation
                if(!p().is_orthogonalised() || p.is_root())
                {
                    const auto& psi = p().as_matrix();
                    CALL_AND_HANDLE(res() = transpose(psi)*psi, "Failed to apply the branch node contraction.");
                }
                is_identity() = p().is_orthogonalised();
            }
        }        
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
    }
};

}   //namespace ttns


#endif  //HTUCKER_MATRIX_ELEMENT_CORE_HPP//

