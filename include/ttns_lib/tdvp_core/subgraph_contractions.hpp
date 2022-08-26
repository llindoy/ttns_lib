#ifndef HTUCKER_MATRIX_ELEMENT_SUBGRAPH_CONTRACTIONS_HPP
#define HTUCKER_MATRIX_ELEMENT_SUBGRAPH_CONTRACTIONS_HPP

#include "../operators/sum_of_product_operator.hpp"
#include "kronecker_product_operator_helper.hpp"

namespace ttns
{

template <typename T, typename B> class matrix_element_base_node_data;
template <typename T, typename B> class matrix_element_node_data;
template <typename T, typename B> class single_particle_operator_node_data;

class subgraph_contraction
{
private:
    template <typename T, typename B> using mat = linalg::matrix<T, B>;
    template <typename T, typename B> using hnode = httensor_node_data<T, B>;
    template <typename T, typename B> using rmat = typename linalg::tensor_slice_traits<linalg::tensor<T, 3, B>, T, 3>::slice_type;
    template <typename T, typename B> using optype = operator_type<T,B>;

    template <typename OpNodeType>
    struct validate_opnode_type : std::false_type{};

    template <typename T, typename B>
    struct validate_opnode_type<tree_node<tree_base<matrix_element_base_node_data<T, B> > > >: std::true_type{};

    template <typename T, typename B>
    struct validate_opnode_type<tree_node<tree_base<matrix_element_node_data<T, B> > > >: std::true_type{};

    template <typename T, typename B>
    struct validate_opnode_type<tree_node<tree_base<single_particle_operator_node_data<T, B> > > >: std::true_type{};

public:
    //the different matrix_element leaf contractions
    template <typename T, typename B>
    static void apply(const hnode<T,B>& p, rmat<T,B> res)
    {
        CALL_AND_HANDLE(res = adjoint(p.as_matrix())*p.as_matrix(), "Failed to evaluate inner product of psi with itself.");
    }

    template <typename T, typename B>
    static void apply(const hnode<T,B>& b, const hnode<T,B>& k, rmat<T,B> res)
    {
        CALL_AND_HANDLE(res = adjoint(b.as_matrix())*k.as_matrix(), "Failed to evaluate inner product of bra and ket.");
    }

    template <typename T, typename B>
    static void apply(const optype<T, B>& op, const hnode<T, B>& p, mat<T, B>& op_act, rmat<T,B> res)
    {
        CALL_AND_HANDLE(op.apply(p.as_matrix(), op_act), "Failed to apply operator to ket.");
        CALL_AND_HANDLE(res = adjoint(p.as_matrix())*op_act, "Failed to apply inner product of bra with action of operator on ket.");
    }

    template <typename T, typename B>
    static void apply(const optype<T, B>& op, const hnode<T, B>& b, const hnode<T, B>& k, mat<T, B>& op_act, rmat<T,B> res)
    {
        CALL_AND_HANDLE(op.apply(k.as_matrix(), op_act), "Failed to apply operator to ket.");
        CALL_AND_HANDLE(res = adjoint(b.as_matrix())*op_act, "Failed to apply inner product of bra with action of operator on ket.");
    }


    //the different matrix_element possible branch contractions
    template <typename T, typename B, typename OpNodeType>
    static void apply(const hnode<T, B>& p, const OpNodeType& opnode, mat<T, B>& op_act, mat<T, B>& temp, rmat<T, B> res)
    {
        static_assert(validate_opnode_type<OpNodeType>::value, "Failed to apply subgraph contraction.  The input operator node is not a valid operator node type.");
        typename B::size_type r = res.slice_index();
        CALL_AND_HANDLE(kronecker_product_operator::apply(opnode, r, p, temp, op_act), "Failed to apply operator to ket.");
        CALL_AND_HANDLE(res = adjoint(p.as_matrix())*op_act, "Failed to apply inner product of bra with action of operator on ket.");
    }

    template <typename T, typename B, typename OpNodeType>
    static void apply(const hnode<T, B>& b, const hnode<T, B>& k, const OpNodeType& opnode, mat<T, B>& op_act, mat<T, B>& temp, rmat<T, B> res)
    {
        static_assert(validate_opnode_type<OpNodeType>::value, "Failed to apply subgraph contraction.  The input operator node is not a valid operator node type.");
        typename B::size_type r = res.slice_index();
        CALL_AND_HANDLE(kronecker_product_operator::apply(opnode, r, k, temp, op_act), "Failed to apply operator to ket.");
        CALL_AND_HANDLE(res = adjoint(b.as_matrix())*op_act, "Failed to apply inner product of bra with action of operator on ket.");
    }
};  //class subgraph_contractions

}   //namespace ttns

#endif

