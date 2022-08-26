#ifndef HTUCKER_TDVP_ALGORITHM_ENGINE_HPP
#define HTUCKER_TDVP_ALGORITHM_ENGINE_HPP

#define TIMING

#include "../utils/timing_macro.hpp"
#include "../utils/zip.hpp"

#include "../httensor.hpp"
#include "../operators/sop_operator.hpp"
#include "../ttn_nodes/operator_node.hpp"


#include "root_to_leaf_decomposition.hpp"
#include "leaf_to_root_decomposition.hpp"
#include "decomposition_engine.hpp"
#include "tdvp_tree_traversal.hpp"


#include "single_particle_operator.hpp"
#include "mean_field_operator.hpp"
#include "operator_container.hpp"

namespace ttns
{

template <template <typename, typename > class Impl, typename T, typename backend = linalg::blas_backend>
class tdvp_algorithm_base
{
protected:
    using impl_type = Impl<T, backend>;
    using vec_type = linalg::vector<T, backend>;
    using mat_type = linalg::matrix<T, backend>;
    using triad_type = std::vector<mat_type>;
    
    using opdata_type = operator_node_data<T, backend>;

    using operator_type = sop_operator<T, backend>;

    using engine_type = decomposition_engine<T, backend, false>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

    using spo_core = single_particle_operator_engine<T, backend>;
    using mfo_core = mean_field_operator_engine<T, backend>;
    using r2l_core = root_to_leaf_decomposition_engine<T, backend>;
    using l2r_core = leaf_to_root_decomposition_engine<T, backend>;

    using dmat_type = typename engine_type::dmat_type;

    using hnode = httensor_node<T, backend>;
    using hdata = httensor_node_data<T, backend>;

    using opnode = typename tree<opdata_type>::node_type;
    using matnode = typename tree<mat_type>::node_type;
public:
    tdvp_algorithm_base() : m_validate_inputs(true) {}
    tdvp_algorithm_base(const httensor<T, backend>& A)  
    {
        m_validate_inputs = true;
        CALL_AND_HANDLE(initialise(A), "Failed to construct tdvp_algorithm_base.");
    }   
    tdvp_algorithm_base(const tdvp_algorithm_base& o) = default;
    tdvp_algorithm_base(tdvp_algorithm_base&& o) = default;

    tdvp_algorithm_base& operator=(const tdvp_algorithm_base& o) = default;
    tdvp_algorithm_base& operator=(tdvp_algorithm_base&& o) = default;
    
    void initialise_base(const httensor<T, backend>& A)
    {
        try
        {   
            bool use_capacity = true;
            ASSERT(A.is_orthogonalised(), "The input hierarchical tucker tensor must have been orthogonalised.");

            size_type maxsize = 0;
            size_type maxcapacity = 0;
            size_type max_cdim2 = 0;
            size_type max_dim2 = 0;
            for(const auto& a : A)
            {
                size_type size = a().size();            if(size > maxsize){maxsize = size;}
                size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}

                size_type dim2i  = a().max_hrank()*a().max_hrank();   if(dim2i > max_cdim2){max_cdim2 = dim2i;}
                dim2i  = a().hrank()*a().hrank();   if(dim2i > max_cdim2){max_cdim2 = dim2i;}
                for(size_type i = 0; i < a.nmodes(); ++i)
                {
                    dim2i  = a().max_dim(i)*a().max_dim(i);   if(dim2i > max_cdim2){max_cdim2 = dim2i;}
                    dim2i  = a().dim(i)*a().dim(i);   if(dim2i > max_dim2){max_dim2 = dim2i;}
                }
            }
            if(!use_capacity)
            {
                maxcapacity = maxsize;
                max_cdim2 = max_dim2;
            }
            if(max_cdim2 > maxcapacity){maxcapacity = max_cdim2;}
#ifdef __NVCC__
            //we need to fix this code, it currently doesn't work
            if(std::is_same<backend, linalg::cuda_backend>::value)
            {
                
                size_type maxmfo = 0;
                for(const auto& a : A)
                {
                    size_type size = mfo_core::contraction_buffer_size(a, use_capacity);   
                    if(size > maxmfo){maxmfo = size;}
                }
                if(maxmfo > maxcapacity){ maxcapacity = maxmfo;}
            }
#endif

            //resize the working arrays.  We will resize these to the maximum possible array sizes
            
            reallocate_working_buffers(maxcapacity);
            resize_working_buffers(1, maxsize);
        
            //set the correct size for all of the objects necessary to compute the node decompositions 
            CALL_AND_HANDLE(m_r.construct_topology(A), "Failed to construct the topology of the r tensor.");
            CALL_AND_HANDLE(m_traversal.resize(A), "Failed to initialise the tree traversal object.");

            for(auto z : zip(m_r, A))
            {
                auto& r = std::get<0>(z);   const auto& a = std::get<1>(z);
                CALL_AND_HANDLE(r2l_core::resize_r_matrix(a(), r(), use_capacity), "Failed to resize elements of the r tensor.");
            }
            try
            {
                m_ortho_engine.template resize<r2l_core>(A, m_u, m_r, use_capacity);
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
            RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
        }
    }

    void clear_base()
    {
        try
        {
            for(size_type i=0; i<m_HA.size(); ++i)
            {
                CALL_AND_HANDLE(m_HA[i].clear(), "Failed to clear a temporary working array tree.");
                CALL_AND_HANDLE(m_temp[i].clear(), "Failed to clear a temporary working array tree.");
                CALL_AND_HANDLE(m_temp2[i].clear(), "Failed to clear a temporary working array tree.");
            }

            CALL_AND_HANDLE(m_temp2.clear(), "Failed to clear a temporary working array tree.");
            CALL_AND_HANDLE(m_HA.clear(), "Failed to clear a temporary working array tree.");
            CALL_AND_HANDLE(m_temp.clear(), "Failed to clear a temporary working array tree.");
            CALL_AND_HANDLE(m_temp2.clear(), "Failed to clear a temporary working array tree.");
            CALL_AND_HANDLE(m_u.clear(), "Failed to clear the u array tree.");
            CALL_AND_HANDLE(m_r.clear(), "Failed to clear the r array tree.");
            CALL_AND_HANDLE(m_ortho_engine.clear(), "Failed to clear the orthogonalisation engine.");
            CALL_AND_HANDLE(m_traversal.clear(), "Failed to clear the traversal engine object.");
            m_nh_evals = 0;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }


    bool operator()(httensor<T, backend>& A, operator_type& op, operator_container<T, backend>& ham)
    {
        try
        {
            impl_type* impl = static_cast<impl_type*>(this);

            return update(A, op, ham, 
                [impl](hnode& _A, opnode& _h, operator_type& _op)
                {
                    CALL_AND_RETHROW(impl->update_node_tensor(_A, _h, _op));
                }, 
                [impl](matnode& _r, opnode& _h)
                {   
                    CALL_AND_RETHROW(impl->update_R_tensor(_r, _h));
                }, 
                [impl](hnode& _a1, hnode& _a2, mat_type& _r, const dmat_type& _pops, opnode& _h, operator_type& _op)
                {
                    CALL_AND_RETHROW(return impl->subspace_expansion_down(_a1, _a2, _r, _pops, _h, _op));
                },
                [impl](hnode& _a1, hnode& _a2, mat_type& _r,const dmat_type& _pops, opnode& _h, operator_type& _op)
                {
                    CALL_AND_RETHROW(return impl->subspace_expansion_up(_a1, _a2, _r, _pops, _h, _op));
                }
            );
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform integration.");
        }
    }

    template <typename NodeFunc, typename RFunc, typename SubspaceFuncDown, typename SubspaceFuncUp>
    bool update(httensor<T, backend>& A, operator_type& op, operator_container<T, backend>& ham, NodeFunc&& nf, RFunc&& rf, SubspaceFuncDown&& sfd, SubspaceFuncUp&& sfu)
    {
        try
        {
            if(m_validate_inputs)
            {   
                ASSERT(A.is_orthogonalised(), "The input hierarchical tucker tensor must be in the orthogonalised form.");
                //ASSERT(ham.validate_size(A, op), "The input hamiltonian container does not have the correct size.");
    
                size_type maxsize = 0;
                for(const auto& a : A)
                {
                    if(a().size() > maxsize){maxsize = a().size();}
                }
                ASSERT(m_HA.size() != 0, "The internal buffers are not large enough to store the temporary results.");
                ASSERT(maxsize <= m_HA[0].capacity(), "The internal buffers are not large enough to store the temporary results.");
            }

            bool subspace_expanded_f = false;

            CALL_AND_HANDLE
            (   
                subspace_expanded_f = forward_loop_step(A, op, ham, std::forward<NodeFunc>(nf), std::forward<RFunc>(rf), std::forward<SubspaceFuncDown>(sfd)), 
                "Failed to perform a step of the tdvp_engine object.  Exception raised when performing the forward loop half step."
            );
            CALL_AND_HANDLE(static_cast<impl_type*>(this)->advance_half_step(), "Failed to advance the implementation specific objects.");
            //if the forward step failed due to a numerical issue we return that it failed.
            bool subspace_expanded_b = false;
            CALL_AND_HANDLE
            (
                subspace_expanded_b = backward_loop_step(A, op, ham, std::forward<NodeFunc>(nf), std::forward<RFunc>(rf), std::forward<SubspaceFuncUp>(sfu)), 
                "Failed to perform a step of the tdvp_engine object.  Exception raised when performing the backward loop half step."
            );
            CALL_AND_HANDLE(static_cast<impl_type*>(this)->advance_half_step(), "Failed to advance the implementation specific objects.");
            A.set_is_orthogonalised();
            return subspace_expanded_f || subspace_expanded_b;
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying the tdvp_algorithm_base to evolve a hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the tdvp_algorithm_base to evolve a hierarchical tucker tensor.");
        }
    }

    bool prepare_evolution(httensor<T, backend>& A, operator_type& op, operator_container<T, backend>& ham)
    {
        
        //first iterate through the tree computing the single particle Hamiltonians.  Here we do not attempt to do any bond dimension adaptation
        //as as we are not time-evolving all information about the optimal unoccupied SHFs will be destroyed before it could be used.
        for(auto z : rzip(A, ham.op()))
        {
            const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
            for(size_type i=0; i < m_HA.size(); ++i)
            {
                CALL_AND_HANDLE(m_HA[i].resize(a().shape(0), a().shape(1)), "Failed to resize HA buffer.");
                CALL_AND_HANDLE(m_temp[i].resize(a().shape(0), a().shape(1)), "Failed to resize temp buffer.");
            }
            if(!a.is_root())
            {
                CALL_AND_HANDLE(spo_core::evaluate(op, a, m_HA, m_temp, hspf), "Failed to evaluate the single particle operator tree nodes.");
            }
        }

        bool subspace_expanded = false;
        impl_type* impl = static_cast<impl_type*>(this);
        
        //we keep attempting the subspace expansion initially until it stops expanding.
        bool attempt_expansion = true;
        while(attempt_expansion)
        {
            bool subspace_expanded_f = false;
            CALL_AND_HANDLE
            (
                subspace_expanded_f = forward_loop_step(A, op, ham, 
                    [](hnode&, opnode&, operator_type&){},
                    [](matnode&, opnode&){},
                    //[](hnode&, hnode&, mat_type&, const dmat_type&, opnode&, operator_type&){return false;}
                    [impl](hnode& _a1, hnode& _a2, mat_type& _r, const dmat_type& _pops, opnode& _h, operator_type& _op)
                    {
                        CALL_AND_RETHROW(return impl->subspace_expansion_down(_a1, _a2, _r, _pops, _h, _op));
                    }
                ),
                "Failed to perform backward branch of expansion step."
            );

            //now perform the backwards loop step but without any evolution of nodes.  Here we apply the subspace expansion in order to
            //construct optimal unoccupied SPFs used for the first step of the evolution
            bool subspace_expanded_b = false;
            CALL_AND_HANDLE
            (
                subspace_expanded_b = backward_loop_step(A, op, ham, 
                    [](hnode&, opnode&, operator_type&){},
                    [](matnode&, opnode&){},
                    [impl](hnode& _a1, hnode& _a2, mat_type& _r, const dmat_type& _pops, opnode& _h, operator_type& _op)
                    {
                        CALL_AND_RETHROW(return impl->subspace_expansion_up(_a1, _a2, _r, _pops, _h, _op));
                    }
                ),
                "Failed to perform subspace expansion step."
            );

            if(subspace_expanded_f || subspace_expanded_b){subspace_expanded = true;}
            if(!subspace_expanded_f && !subspace_expanded_b){attempt_expansion = false;}
            //subspace_expanded = subspace_expanded_b;
            //attempt_expansion = subspace_expanded_b;
            A.set_is_orthogonalised();
        }
        return subspace_expanded;
    }

protected:  
    bool subspace_expansion_down(hnode& , hnode&, mat_type& , const dmat_type&, opnode& , operator_type&){return false;}
    bool subspace_expansion_up(hnode& , hnode&, mat_type& , const dmat_type&, opnode& , operator_type&){return false;}

    void update_node_tensor(hnode&, opnode&, operator_type&){}
    void update_R_tensor(matnode&, opnode&){}

    template <typename NodeFunc, typename RFunc, typename SubspaceFunc>
    bool forward_loop_step(httensor<T, backend>& psi, operator_type& op, operator_container<T, backend>& ham, NodeFunc&& nf, RFunc&& rf, SubspaceFunc&& sf)
    {
        try
        {       
            bool subspace_expanded = false;
            auto& m_ham = ham.op();
            m_traversal.reset_times_visited();
        
            //traverse the tree in the order specified by m_traversal
            for(size_type id : m_traversal)
            {
                //define aliases for all of the arguments at the current node
                auto& A = psi[id];          auto& u = m_u;          auto& r = m_r[id];
                auto& h = m_ham[id];       

                CALL_AND_HANDLE(resize_working_buffers(A().shape(0), A().shape(1)), "Failed to resize working buffers.");

                size_type times_visited = m_traversal.times_visited(id);
                ++(m_traversal.times_visited(id));

                //now provided this isn't the first time we've traversed the node we will need to apply a root to leaf node decomposition to 
                //it so that we can propagate factors down the tree structure to its children.
                if(times_visited != A.nmodes())
                {
                    //get the index of the child we will be performing the decomposition for
                    size_type mode = times_visited;

                    //if it is our first time visiting the node and we are not at the root node we need to apply the parent nodes root to leaf decomposition
                    if(times_visited == 0 && !A.is_root())
                    {
                        //now we apply the parents r matrix to this node
                        CALL_AND_HANDLE(r2l_core::apply_from_parent(A, r, m_temp[0]), "Failed to apply the parents root to leaf decomposition to the current node.");
                    }

                    //evaluate the root to leaf decomposition provided we aren't at the leaf node and update the mean field hamiltonian
                    if(!A.is_leaf())
                    {
                        CALL_AND_HANDLE(r2l_core::evaluate(m_ortho_engine, A, u, r, m_temp[0], mode), "Failed to compute the root to leaf decomposition for a node.");
                        CALL_AND_HANDLE(r2l_core::apply_to_node(A, u), "Failed to apply the result of the root to leaf decomposition to the current node.");    

                        bool seloc = false;
                        //as we descend the tree apply the subspace expansion
                        CALL_AND_HANDLE(seloc = sf(A[mode], A, r(), m_ortho_engine.S(), h[mode], op), "Subspace expansion Failed.");
                        if(seloc){subspace_expanded = true;}
                        //now we can update the mean field Hamiltonian at the node.  
                        CALL_AND_HANDLE(mfo_core::evaluate(A, m_HA, m_temp, m_temp2, h[mode]), "Failed to evaluate the mean field operator.");
                    }
                }
                //if it is our final time accessing the node we need to perform the evolution steps
                else
                {
                    CALL_AND_HANDLE(nf(A, h, op), "Failed to update node tensor.");

                    //now provided this node is not the root we evaluate the leaf to root decomposition of this tensor
                    //time evolve the coefficient tensor and apply it to its parent
                    if(!A.is_root())
                    {
                        //modify the l2r_core::evaluate routine to include subspace expansion of the child node
                        CALL_AND_HANDLE(l2r_core::evaluate(m_ortho_engine, A, u, r), "Failed to evaluate the leaf_to_root_decomposition for a given node.");

                        CALL_AND_HANDLE(l2r_core::apply_to_node(A, u), "Failed when applying the result of the leaf_to_root_decomposition to the current node.");

                        //now we can update the single particle Hamiltonian at the node
                        CALL_AND_HANDLE(spo_core::evaluate(op, A, m_HA, m_temp, h), "Failed to evaluate the single particle operator");
    
                        CALL_AND_HANDLE(rf(r, h), "Failed to update R coefficient tensor.");

                        CALL_AND_HANDLE(l2r_core::apply_to_parent(A, r, m_temp[0]), "Failed when applying the result of the leaf_to_root_decomposition to the parent node.");
                    }
                }
            } 
            return subspace_expanded;
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying forward half step of the tdvp_algorithm_base.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply forward half step of the tdvp_algorithm_base.");
        }
    }

    template <typename NodeFunc, typename RFunc, typename SubspaceFunc>
    bool backward_loop_step(httensor<T, backend>& psi, operator_type& op, operator_container<T, backend>& ham, NodeFunc&& nf, RFunc&& rf, SubspaceFunc&& sf)
    {
        try
        {
            bool subspace_expanded = false;
            auto& m_ham = ham.op();
            m_traversal.reset_times_visited();
            
            //traverse the tree in the reverse of the order specified by m_traversal
            for(size_type id : reverse(m_traversal))
            {
                //define aliases for all of the arguments at the current node
                auto& A = psi[id];          auto& u = m_u;          auto& r = m_r[id];
                auto& h = m_ham[id];       
                
                CALL_AND_HANDLE(resize_working_buffers(A().shape(0), A().shape(1)), "Failed to resize working buffers.");

                size_type times_visited = m_traversal.times_visited(id);
                ++(m_traversal.times_visited(id));

                //if it is not our last time visiting the node we only need to update the single hole decomposition and mean field Hamiltonian
                if(times_visited != A.nmodes())
                {
                    //get the index of the child we will be performing the decomposition for
                    size_type mode = A.nmodes() - (times_visited+1);

                    //now if this is the first time we have accessed this node we firt need to apply its parent's decomposition, 
                    //which is first backwards time evolved.  Following which we can time evolve this nodes coefficient matrix.
                    if(times_visited == 0)
                    {
                        //we only have a parent node if we aren't at the root node.
                        if(!A.is_root())
                        {
                            //time evolve the parents r matrix backwards in time through half a time step using this nodes representation of the full Hamiltonian
                            CALL_AND_HANDLE(rf(r.parent(), h), "Failed to update R coefficient tensor.");

                            //now we apply the parents r matrix to this node
                            CALL_AND_HANDLE(r2l_core::apply_from_parent(A, r, m_temp[0]), "Failed to apply the parents root to leaf decomposition to the current node.");
                        }

                        CALL_AND_HANDLE(nf(A, h, op), "Failed to update node tensor.");
                    }

                    //now provided this node isn't a leaf node we need to evaluate its root to leaf decomposition so that we can apply this result
                    //to its children.  Upon doing so we can now update the mean field operators at this node
                    if(!A.is_leaf())
                    {
                        CALL_AND_HANDLE(r2l_core::evaluate(m_ortho_engine, A, u, r, m_temp[0], mode), "Failed to compute the root to leaf decomposition for a node.");
                        CALL_AND_HANDLE(r2l_core::apply_to_node(A, u), "Failed to apply the result of the root to leaf decomposition to the current node.");
                        CALL_AND_HANDLE(mfo_core::evaluate(A, m_HA, m_temp, m_temp2, h[mode]), "Failed to evaluate the mean field operator.");
                    }
                }
                //on the final time accessing we need to apply the leaf to root decomposition to construct the new single particle functions at this node.
                else 
                {
                    //in the backwards loop we attempt to expand the bond dimension when moving up the tree
                    if(!A.is_root())
                    {
                        CALL_AND_HANDLE(l2r_core::evaluate(m_ortho_engine, A, u, r), "Failed to evaluate the leaf_to_root_decomposition for a given node.");
                        CALL_AND_HANDLE(l2r_core::apply_to_node(A, u), "Failed when applying the result of the leaf_to_root_decomposition to the current node.");

                        bool seloc = false;
                        CALL_AND_HANDLE(seloc = sf(A, A.parent(), r(), m_ortho_engine.S(), h, op), "Subspace expansion failed.");
                        if(seloc){subspace_expanded = true;}

                        CALL_AND_HANDLE(l2r_core::apply_to_parent(A, r, m_temp[0]), "Failed when applying the result of the leaf_to_root_decomposition to the parent node.");

                        //now we can update the single particle Hamiltonian at the node
                        CALL_AND_HANDLE(spo_core::evaluate(op, A, m_HA, m_temp, h), "Failed to evaluate the single particle operator");
                    }
                }
            }
            return subspace_expanded;
        }
        catch(const linalg::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying backward half step of the tdvp_algorithm_base.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply backward half step of the tdvp_algorithm_base.");
        }
    }

    bool& validate_inputs(){return m_validate_inputs;}
    const bool& validate_inputs() const {return m_validate_inputs;}

protected:
    void reallocate_working_buffers(size_t maxcapacity)
    {
        size_type nbuffers = static_cast<impl_type*>(this)->nbuffers();
        CALL_AND_HANDLE(m_HA.resize(nbuffers), "Failed to resize the opA array.");
        CALL_AND_HANDLE(m_temp.resize(nbuffers), "Failed to resize the temporary matrix.");
        CALL_AND_HANDLE(m_temp2.resize(nbuffers), "Failed to resize the temporary matrix.");
        for(size_type i=0; i<m_HA.size(); ++i)
        {
            CALL_AND_HANDLE(m_HA[i].reallocate(maxcapacity), "Failed to resize the opA array.");
            CALL_AND_HANDLE(m_temp[i].reallocate(maxcapacity), "Failed to resize the temporary matrix.");
            CALL_AND_HANDLE(m_temp2[i].reallocate(maxcapacity), "Failed to reszie temporary matrix.");
        }

        CALL_AND_HANDLE(m_u.reallocate(maxcapacity), "Failed to resize u tensor.");
    }

    void resize_working_buffers(size_type s1, size_type s2)
    {
        for(size_type i=0; i<m_HA.size(); ++i)
        {
            CALL_AND_HANDLE(m_HA[i].resize(s1, s2), "Failed to resize the opA array.");
            CALL_AND_HANDLE(m_temp[i].resize(s1, s2), "Failed to resize the temporary matrix.");
            CALL_AND_HANDLE(m_temp2[i].resize(s1, s2), "Failed to reszie temporary matrix.");
        }
        CALL_AND_HANDLE(m_u.resize(s1, s2), "Failed to resize u tensor.");
    }

public:
    size_type nh_applications() const{return m_nh_evals;}

protected:
    mutable triad_type m_HA;
    mutable triad_type m_temp;
    mutable triad_type m_temp2;
    
    mat_type m_u;
    tree<mat_type> m_r;
    size_type m_nh_evals;

    //an object wrapping the decomposition engine.  This also stores the workspace arrays required for the computations.
    engine_type m_ortho_engine;

    //an object storing the traversal order required for evaluating the root to leaf decomposition
    tdvp_tree_traversal m_traversal;

    size_type m_num_threads;
    
    bool m_validate_inputs;

};  //class tdvp_algorithm_base

}   //namespace ttns

#endif  //HTUCKER_TDVP_ALGORITHM_ENGINE_HPP//

